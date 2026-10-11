"""Native format/codec fixtures from development parents, not accuracy tests."""
from fractions import Fraction
import json
from pathlib import Path
import sys
import av
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT,atomic_json,sha256
from backend.media import read_image,sample_video
from scripts.evaluate_public_images_live import freeze
from backend.contracts import load_config

WORK=ROOT/'data/format_matrix_v1.4'


def main():
    WORK.mkdir(exist_ok=True);rows=[]
    image_rows=json.loads((ROOT/'data/public_images_v1.4/manifest.json').read_text('utf-8'))
    parent=next(r for r in image_rows if r['split']=='fusion' and r['truth_label']=='genuine')
    photo,_=read_image(ROOT/'data/public_images_v1.4'/parent['relative_path'],50000000);photo=photo.resize((320,240))
    for name,fmt in [('jpeg','JPEG'),('png','PNG'),('webp','WEBP'),('bmp','BMP'),('tiff','TIFF'),('avif','AVIF'),('heic','HEIF'),('heif','HEIF')]:
        path=WORK/('native.'+name);kwargs={'quality':90} if fmt in ('JPEG','WEBP','AVIF','HEIF') else {}
        if not path.exists():photo.save(path,format=fmt,**kwargs)
        rows.append({'sample_id':'format_'+name,'material_type':'image','workspace_relative_path':path.relative_to(ROOT).as_posix(),
          'sha256':sha256(path),'expected_format':fmt,'parent_sha256':parent['sha256'],'scope':'Reencoded development photo; compatibility only'})
    rgba=WORK/'alpha.png'
    if not rgba.exists():
        image=photo.convert('RGBA');alpha=np.full((240,320),255,np.uint8);alpha[:60,:80]=0;image.putalpha(Image.fromarray(alpha));image.save(rgba)
    rows.append({'sample_id':'format_rgba','material_type':'image','workspace_relative_path':rgba.relative_to(ROOT).as_posix(),'sha256':sha256(rgba),'expected_format':'PNG'})
    oriented=WORK/'oriented.heic'
    if not oriented.exists():
        exif=Image.Exif();exif[274]=6;exif[305]='known format QA encoder';photo.save(oriented,format='HEIF',exif=exif,quality=90)
    rows.append({'sample_id':'format_heic_orientation','material_type':'image','workspace_relative_path':oriented.relative_to(ROOT).as_posix(),
      'sha256':sha256(oriented),'expected_format':'HEIF','orientation_metadata_expected':True})
    video_rows=json.loads((ROOT/'data/public_genvideo_v1.4/manifest.json').read_text('utf-8'))
    vparent=next(r for r in video_rows if r['split']=='train' and r['truth_label']=='genuine')
    with av.open(str(ROOT/'data/public_genvideo_v1.4'/vparent['relative_path'])) as source:
        originals=[]
        for frame in source.decode(video=0):
            originals.append(frame.to_image().convert('RGB').resize((320,240)))
            if len(originals)==24:break
    if len(originals)<24:raise ValueError('Format development parent has fewer than24 frames; do not silently manufacture motion')
    variants=[('h264_mp4','.mp4','libx264','yuv420p'),('h264_mov','.mov','libx264','yuv420p'),
      ('h264_mkv','.mkv','libx264','yuv420p'),('hevc_mp4','.mp4','libx265','yuv420p'),
      ('hevc_mov','.mov','libx265','yuv420p'),('hevc10_mp4','.mp4','libx265','yuv420p10le'),
      ('vp9_webm','.webm','libvpx-vp9','yuv420p'),('av1_mp4','.mp4','libsvtav1','yuv420p'),
      ('av1_webm','.webm','libsvtav1','yuv420p'),('mpeg4_avi','.avi','mpeg4','yuv420p'),
      ('mjpeg_avi','.avi','mjpeg','yuvj420p'),('prores_mov','.mov','prores_ks','yuv422p10le')]
    for name,suffix,codec,pixel_format in variants:
        path=WORK/(name+suffix)
        if not path.exists():
            with av.open(str(path),'w') as container:
                stream=container.add_stream(codec,rate=24);stream.width=320;stream.height=240;stream.pix_fmt=pixel_format
                stream.codec_context.thread_count=2
                if codec=='libx264':stream.options={'preset':'ultrafast','crf':'20','bf':'0'}
                if codec=='libx265':stream.options={'preset':'ultrafast','crf':'22','x265-params':'log-level=error:pools=2:frame-threads=2'}
                if codec=='libvpx-vp9':stream.options={'deadline':'realtime','cpu-used':'8'}
                if codec=='libsvtav1':stream.options={'preset':'12','svtav1-params':'lp=2'}
                for index,image in enumerate(originals):
                    frame=av.VideoFrame.from_image(image);frame.pts=index;frame.time_base=Fraction(1,24)
                    for packet in stream.encode(frame):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
        frames,_,meta=sample_video(path,load_config())
        if not frames or any(f.pts is None for f in frames):raise ValueError('No actual decoded PTS')
        rows.append({'sample_id':'format_'+name,'material_type':'video','workspace_relative_path':path.relative_to(ROOT).as_posix(),
          'sha256':sha256(path),'expected_codec':codec.replace('libx','h').replace('libvpx-vp9','vp9').replace('libsvtav1','av1'),
          'requested_pixel_format':pixel_format,'decoded_codec':meta['tracks'][0]['codec'],'parent_sha256':vparent['sha256'],
          'scope':'Transcoded real-source development prefix; no independent accuracy claim'})
    freeze(WORK/'manifest.json',rows)
    atomic_json(ROOT/'runs/format_decode_v1.4.json',{'fixtures':len(rows),'original_source_files_unchanged':True,
      'image_extensions':8,'video_container_codec_pairs':12,'decoding_verified':True,'full_pipeline_verified':False,
      'formal_accuracy_test':False,'manifest_sha256':sha256(WORK/'manifest.json')})
    print('Generated and decoded native format fixtures',len(rows),flush=True)


if __name__=='__main__':main()
