"""CARLA spectator-follow integration showcase; recorded-data VISUAL RECONSTRUCTION.

This is a presentation tool, not a planning/tracking experiment. It reads saved
experiment CSVs and TELEPORTS a physics-disabled prop vehicle through their
recorded CommonRoad poses. It NEVER applies throttle/brake, and it NEVER claims
that offline planned trajectories were physically driven. Only this process
may tick the synchronous Town01 world. See README_SHOWCASE.md before running.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import showcase_data_m6 as d

FPS=20
DT=.05
DEFAULT_ROOT=Path(__file__).resolve().parent.parent
DEFAULT_OUT=Path(__file__).resolve().parent/'output/carla_showcase/integration_showcase_001'


def lookup_scene_index(scene, data, t):
    """Keep archived sample selection accurate, even for shorter aborted runs."""
    samples=data[scene]
    if not 0<=t<=float(samples[-1]['t_s'])+.051:
        raise ValueError('Playback time outside source archive')
    return min(len(samples)-1, max(0,round(t/DT)))


def scene_pose(name,data,idx):
    if name in ('straight','steer_pos','steer_neg','route_off','route_on','speed_failed','speed_fixed'):
        return d.pose(data[name][idx])
    if name=='feedback':
        return d.pose(data['empirical_on'][min(idx,len(data['empirical_on'])-1)])
    if name=='planning':
        return d.observation_pose(data['observation'])
    if 'observation' in data:
        return d.observation_pose(data['observation'])
    return None


def location_and_yaw(carla, p, z):
    # CommonRoad right-handed y and yaw are mirrored relative to CARLA.
    return carla.Transform(carla.Location(x=p.x,y=-p.y,z=z),
                           carla.Rotation(yaw=-math.degrees(p.yaw)))


def actor_origin(carla, center_pose, z, bounding_box):
    yaw=-center_pose.yaw
    bx=float(bounding_box.location.x)
    by=float(bounding_box.location.y)
    cx,cy=center_pose.x,-center_pose.y
    x=cx-bx*math.cos(yaw)+by*math.sin(yaw)
    y=cy-bx*math.sin(yaw)-by*math.cos(yaw)
    return carla.Transform(carla.Location(x=x,y=y,z=z),
                           carla.Rotation(yaw=-math.degrees(center_pose.yaw)))


def camera_transform(carla, target, *, base_z, wide=False):
    yaw=-target.yaw
    if wide:
        back,side,height,pitch=9.,0.,29.,-77.
    else:
        back,side,height,pitch=7.5,1.5,4.4,-18.
    x=target.x-back*math.cos(yaw)-side*math.sin(yaw)
    y=-target.y-back*math.sin(yaw)+side*math.cos(yaw)
    return carla.Transform(carla.Location(x=x,y=y,z=base_z+height),
                           carla.Rotation(yaw=math.degrees(yaw),pitch=pitch))


def halfway(a,b):
    dx=b.x-a.x;dy=b.y-a.y
    return d.Pose((a.x+b.x)/2,(a.y+b.y)/2,a.yaw)


def collision_free_preflight(carla, world, bundles):
    """No live actors / other tick owner. Kinematic ghost is NOT crash-safe driving."""
    scene=world.get_map().name
    if scene.split('/')[-1]!='Town01':
        raise RuntimeError(f'Expected Town01, found {scene}')
    if bool(world.get_settings().synchronous_mode):
        raise RuntimeError('World is already synchronous; do not take over another tick owner')
    actors=world.get_actors()
    vehicles=[a for a in actors if a.type_id.startswith('vehicle.')]
    walkers=[a for a in actors if a.type_id.startswith('walker.')]
    sensors=[a for a in actors if a.type_id.startswith('sensor.')]
    if vehicles or walkers or sensors:
        raise RuntimeError(f'Town01 is not isolated: vehicles={len(vehicles)}, walkers={len(walkers)}, sensors={len(sensors)}')
    # Town01 has many ordinary static.prop.mesh actors. They do not prevent
    # visualization, but count/log them (no deletion or manipulation).
    props=[a for a in actors if a.type_id.startswith('static.prop.')]
    return {'map':scene,'vehicles':len(vehicles),'walkers':len(walkers),
            'sensors':len(sensors),'existing_static_props':len(props),
            'synchronous_mode_before':False,
            'note':'Kinematic presentation only; no physics or live-drive safety certification'}


def draw_path(world,carla,rows,color,sim_lifetime_s,*, planning=False):
    # CARLA world.debug is an on-screen overlay; video also gets an inset chart.
    pts=[d.pose(r,planning=planning) for r in rows]
    for p,q in zip(pts[:-1],pts[1:]):
        if abs(p.x-q.x)+abs(p.y-q.y)<.0002:continue
        x,y=p.x,-p.y
        try:
            z=world.get_map().get_waypoint(carla.Location(x=x,y=y,z=1.)).transform.location.z+.22
        except Exception:
            z=.22
        a=carla.Location(x,y,z)
        b=carla.Location(q.x,-q.y,z)
        world.debug.draw_line(a,b,thickness=.13,color=color,life_time=sim_lifetime_s,persistent_lines=False)


def seek_rows_for_plot(data,name):
    if name=='planning':
        return [(data[k],label,col,True) for k,label,col in [('plan_safety','Safety / R9-first','#62e6ad'),('plan_corridor','Corridor / R5-first','#ffbd60')] if k in data]
    if name=='feedback':
        return [(data[k],label,col,False) for k,label,col in [('empirical_off','Feedback off','#ffbd60'),('empirical_on','Feedback on','#62e6ad')] if k in data]
    if name in ('route_off','route_on'):
        return [(data[k],label,col,False) for k,label,col in [('route_off','Feedback off','#ffbd60'),('route_on','Feedback on','#62e6ad')] if k in data]
    if name in ('speed_failed','speed_fixed'):
        return [(data[k],label,col,False) for k,label,col in [('speed_failed','Aborted run','#ff717d'),('speed_fixed','Coast fix','#62e6ad')] if k in data]
    if name in ('steer_pos','steer_neg'):
        return [(data[k],label,col,False) for k,label,col in [('steer_pos','Steer +0.08','#ffbd60'),('steer_neg','Steer -0.08','#62e6ad')] if k in data]
    return []


def to_png_chart(draw, font, rowspec, rect, name):
    """A compact guaranteed-visible plot, sourced from archived CSV rows."""
    x0,y0,x1,y1=rect
    draw.rounded_rectangle((x0,y0,x1,y1),radius=12,fill=(11,23,39,218))
    mx=x0+28;my=y0+27;rx=x1-14;ry=y1-33
    ys=[];xs=[];series=[]
    for rows,label,col,planning in rowspec:
        coords=[]
        for r in rows:
            if name=='planning' or name in ('steer_pos','steer_neg'):
                p=d.pose(r,planning=planning)
                coords.append((p.x,p.y))
            elif name in ('speed_failed','speed_fixed'):
                coords.append((d.number(r['t_s'],'time'), d.number(r['forward_speed_mps'],'speed')))
            else:
                coords.append((d.number(r['t_s'],'time'),d.number(r['position_error_m'],'error')))
        series.append((coords,label,col))
        xs.extend(a for a,b in coords);ys.extend(b for a,b in coords)
    if not xs:return
    xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys)
    if xmax-xmin<.001:xmax=xmin+.001
    if ymax-ymin<.001:ymax=ymin+.001
    dx=xmax-xmin;dy=ymax-ymin
    xmin-=.12*dx;xmax+=.12*dx;ymin-=.12*dy;ymax+=.12*dy
    draw.rectangle((mx,my,rx,ry),outline=(108,128,150,170),width=1)
    for coords,label,col in series:
        pts=[(mx+(x-xmin)/(xmax-xmin)*(rx-mx),ry-(y-ymin)/(ymax-ymin)*(ry-my)) for x,y in coords]
        if len(pts)>1:draw.line(pts,fill=col,width=3,joint='curve')
    for i,(_,label,col) in enumerate(series):
        draw.rectangle((x0+20+i*148,y1-20,x0+31+i*148,y1-12),fill=col)
        draw.text((x0+36+i*148,y1-23),label,font=font,fill='#ffffff')
    hdr=('Actual trajectory positions (CR XY)' if name in ('planning','steer_pos','steer_neg')
         else 'Measured speed (m/s)' if name in ('speed_failed','speed_fixed') else 'Tracking error (m)')
    draw.text((x0+16,y0+5),hdr,font=font,fill='#e7eef5')


def font(size,bold=False):
    from PIL import ImageFont
    sources=([r'C:\Windows\Fonts\segoeuib.ttf',r'C:\Windows\Fonts\arialbd.ttf'] if bold else
             [r'C:\Windows\Fonts\segoeui.ttf',r'C:\Windows\Fonts\arial.ttf'])
    sources+=['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for name in sources:
        try: return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()


def render_caption(image, name, scene_idx, scene_count, frame_number, data, source_paths):
    from PIL import Image,ImageDraw
    canvas=Image.frombytes('RGBA',(int(image.width),int(image.height)), bytes(image.raw_data),'raw','BGRA')
    overlay=Image.new('RGBA',canvas.size,(0,0,0,0));draw=ImageDraw.Draw(overlay)
    w,h=canvas.size
    title,sub,qualification=d.scene_summary(name,data)
    big=font(max(18,int(w*.027)),True);medium=font(max(12,int(w*.017)));small=font(max(11,int(w*.013)))
    draw.rounded_rectangle((18,16,w-18,110),radius=12,fill=(11,23,39,221))
    draw.text((38,23),f'{scene_idx:02d}/{scene_count:02d}  {title}',font=big,fill='#ffffff')
    draw.text((39,72),sub,font=medium,fill='#cfe3fa')
    if name in ('planning','feedback','route_off','route_on','speed_failed','speed_fixed','steer_pos','steer_neg'):
        specs=seek_rows_for_plot(data,name)
        to_png_chart(draw,small,specs,(w-385,125,w-19,325),name)
    draw.rounded_rectangle((18,h-88,w-18,h-16),radius=10,fill=(11,23,39,230))
    draw.text((36,h-81),qualification,font=medium,fill='#e7eef5')
    # Use parent directory as well: nearly all archival CSVs are named
    # calibration_samples.csv, so the bare name would be ambiguous.
    names=[Path(source_paths[k]).parent.name+'/'+Path(source_paths[k]).name
           for k in d.SCENE_KEYS[name] if k in source_paths and k not in ('observation','plan_report')]
    if not names and 'observation' in source_paths:
        p=Path(source_paths['observation'])
        names=[p.parent.name+'/'+p.name]
    for j,item in enumerate(names[:2]):
        draw.text((36,h-55+j*17),('SOURCE: ' if j==0 else '           ')+item,
                  font=small,fill='#9cb5cd')
    if len(names)>2:
        draw.text((w-190,h-35),f'+{len(names)-2} sources in report',font=small,fill='#9cb5cd')
    final=Image.alpha_composite(canvas,overlay).convert('RGB')
    return final.tobytes()


class Film:
    def __init__(self, path, width, height, slowdown):
        import imageio_ffmpeg
        self.path=path; self.width=width;self.height=height;self.slowdown=slowdown
        self._gen=imageio_ffmpeg.write_frames(str(path),(width,height),fps=FPS,
            codec='libx264',pix_fmt_out='yuv420p',output_params=['-movflags','+faststart'],
            ffmpeg_log_level='warning')
        self._gen.send(None)
        self.frames=0
    def send(self, data):
        for i in range(self.slowdown):
            self._gen.send(data)
            self.frames+=1
    def close(self):
        if self._gen is not None:
            self._gen.close();self._gen=None


def scene_frame_count(name,data):
    if name in ('intro','outro','observation','planning','feedback'):
        return round(d.scene_seconds(name,data)/DT)
    return len(data[name])


def run(args,bundle,*, carla):
    data=bundle['data']; scenes=bundle['scenes'];paths=bundle['paths']
    client=carla.Client(args.host,args.port);client.set_timeout(args.timeout)
    world=client.get_world()
    preflight=collision_free_preflight(carla,world,bundle)
    if (client.get_server_version().strip()!='0.9.16' or
        client.get_client_version().strip()!='0.9.16'):
        raise RuntimeError('This archived showcase was validated for CARLA 0.9.16')
    out=args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory already contains files; choose a fresh --out')
    out.mkdir(parents=True,exist_ok=True)
    camera=ego=cone=None
    q=queue.Queue()
    spectator=world.get_spectator()
    original_spectator=spectator.get_transform()
    original_settings=world.get_settings()
    settings_changed=False
    film=None
    carla_events=[]
    report={'status':'aborted','execution_mode':'kinematic_telemetry_reconstruction_no_live_driving',
            'original_rulebook_executed':False,'scenic_executed':False,
            'offline_planner_paths_physically_executed':False,
            'CARLA_throttle_brake_commands_issued':0,
            'source_sha256':bundle['source_hashes'],
            'source_paths':{k:str(paths[k]) for k in data},
            'scene_order':list(scenes),'preflight':preflight,
            'video_fps':FPS,'slowdown':args.slowdown,'capture_size':[args.width,args.height],
            'cleanup':{},'notes':['Vehicles are moved kinematically to archived positions with physics disabled.',
            'Offline planned trajectories are only rendered as overlays; no physical obstacle avoidance.',
            'Changing playback rate does not change archival CARLA simulation timestamps or vehicle dynamics.']}
    try:
        import PIL
        import imageio_ffmpeg
        # Lock world ownership only after all archives and dependencies are verified.
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=DT
        settings.no_rendering_mode=False
        settings_changed=True
        world.apply_settings(settings)
        first=(d.observation_pose(data['observation']) if 'observation' in data else
                d.pose(next(iter(data.values()))[0]))
        wp=world.get_map().get_waypoint(carla.Location(first.x,-first.y,z=1.))
        road_z=wp.transform.location.z
        bp=world.get_blueprint_library().find('vehicle.tesla.model3')
        if bp.has_attribute('role_name'): bp.set_attribute('role_name','presentation_only_kinematic')
        ego=world.try_spawn_actor(bp,location_and_yaw(carla,first,road_z+.12))
        if ego is None:raise RuntimeError('CARLA could not spawn temporary visual vehicle')
        ego.set_autopilot(False);ego.set_simulate_physics(False)
        ego.set_transform(actor_origin(carla,first,road_z+.12,ego.bounding_box))
        camera_bp=world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x',str(args.width))
        camera_bp.set_attribute('image_size_y',str(args.height))
        camera_bp.set_attribute('fov','85')
        camera_bp.set_attribute('sensor_tick','0.0')
        first_cam=camera_transform(carla,first,base_z=road_z,wide=True)
        camera=world.spawn_actor(camera_bp,first_cam)
        if camera is None:raise RuntimeError('Could not spawn RGB camera')
        camera.listen(q.put)
        film=Film(out/'showcase.mp4',args.width,args.height,args.slowdown)
        frame_i=0;total_frames=sum(scene_frame_count(x,data) for x in scenes)
        for scene_i,scene in enumerate(scenes,1):
            count=scene_frame_count(scene,data)
            print(f'[{scene_i}/{len(scenes)}] {scene}: {count} simulation frames')
            # The cone appears only in the observation/planning scenes; all
            # recorded driving trials actually had no obstacle actor.
            if scene=='observation' and 'observation' in data and cone is None:
                ob=data['observation']['obstacle']
                cx,cy=ob['center_cr_xy_m'];cone_bp=world.get_blueprint_library().find('static.prop.constructioncone')
                where=d.Pose(float(cx),float(cy),0.)
                cone_wp=world.get_map().get_waypoint(carla.Location(where.x,-where.y,1.))
                cz=cone_wp.transform.location.z+.04
                cone=world.try_spawn_actor(cone_bp,location_and_yaw(carla,where,cz))
                if cone is None:raise RuntimeError('Could not spawn temporary visual cone')
                cone.set_transform(actor_origin(carla,where,cz,cone.bounding_box))
            if scene not in ('intro','observation','planning') and cone is not None:
                ok=bool(cone.destroy());cone=None
                if not ok:raise RuntimeError('Failed to remove visual cone before obstacle-free recorded runs')
            if scene=='planning':
                lifetime=count*DT+.08
                draw_path(world,carla,data['plan_safety'],carla.Color(35,240,153),lifetime,planning=True)
                draw_path(world,carla,data['plan_corridor'],carla.Color(250,166,55),lifetime,planning=True)
            if scene in ('straight','steer_pos','steer_neg','feedback','route_off','route_on','speed_failed','speed_fixed'):
                key=scene if scene in data else 'empirical_on'
                if key in data:
                    draw_path(world,carla,data[key],carla.Color(38,185,241),count*DT+.08)
            for i in range(count):
                time_start=time.monotonic()
                t=i*DT
                p=scene_pose(scene,data,min(i,len(data[scene])-1) if scene in data else i)
                if p is None:p=first
                ego.set_transform(actor_origin(carla,p,road_z+.12,ego.bounding_box))
                if scene=='planning':
                    # Moving DEBUG DOTS are presentation cursors on archived
                    # offline plans, never simulated driven vehicles.
                    for key,col in [('plan_safety',carla.Color(35,240,153)),
                                    ('plan_corridor',carla.Color(250,166,55))]:
                        idx=round(i/max(1,count-1)*(len(data[key])-1))
                        dot=d.pose(data[key][idx],planning=True)
                        world.debug.draw_point(carla.Location(dot.x,-dot.y,road_z+.5),
                            size=.35,color=col,life_time=DT*1.5,persistent_lines=False)
                wide=scene in ('intro','observation','planning','feedback','outro')
                cam_target=p
                if scene in ('intro','observation','planning','outro') and 'observation' in data:
                    cam_target=d.observation_pose(data['observation'])
                    if scene in ('observation','planning'):
                        oc=data['observation']['obstacle']['center_cr_xy_m']
                        cam_target=halfway(cam_target,d.Pose(float(oc[0]),float(oc[1]),cam_target.yaw))
                ct=camera_transform(carla,cam_target,base_z=road_z,wide=wide)
                camera.set_transform(ct)
                spectator.set_transform(ct)
                fid=int(world.tick(args.timeout))
                while True:
                    img=q.get(timeout=args.timeout)
                    if int(img.frame)>=fid:break
                if int(img.frame)!=fid:
                    raise RuntimeError(f'Lost synchronized RGB camera frame: expected {fid}, got {img.frame}')
                frame_data=render_caption(img,scene,scene_i,len(scenes),frame_i,data,paths)
                film.send(frame_data);frame_i+=1
                if args.realtime:
                    elapsed=time.monotonic()-time_start
                    delay=DT*args.slowdown-elapsed
                    if delay>0:time.sleep(delay)
            carla_events.append({'scene':scene,'frames':count,'source_files':[str(paths[k]) for k in d.SCENE_KEYS[scene] if k in paths]})
        report['status']='completed'
        report['scenes_rendered']=carla_events
        report['encoded_video_frames']=film.frames
        report['estimated_video_duration_s']=film.frames/FPS
    except KeyboardInterrupt:
        report['abort_reason']='Interrupted by user'
    except Exception as e:
        report['abort_reason']=type(e).__name__+': '+str(e)
    finally:
        if film is not None:
            try:film.close();report['cleanup']['encoder_closed']=True
            except Exception as e:report['cleanup']['encoder_error']=str(e)
        if camera is not None:
            try:camera.stop();report['cleanup']['camera_destroyed']=bool(camera.destroy())
            except Exception as e:report['cleanup']['camera_error']=str(e)
        else:report['cleanup']['camera_destroyed']=True
        if cone is not None:
            try:report['cleanup']['cone_destroyed']=bool(cone.destroy())
            except Exception as e:report['cleanup']['cone_error']=str(e)
        else:report['cleanup']['cone_destroyed']=True
        if ego is not None:
            try:report['cleanup']['ego_destroyed']=bool(ego.destroy())
            except Exception as e:report['cleanup']['ego_error']=str(e)
        else:report['cleanup']['ego_destroyed']=True
        try:
            spectator.set_transform(original_spectator)
            report['cleanup']['spectator_restored']=True
        except Exception as e: report['cleanup']['spectator_restore_error']=str(e)
        if settings_changed:
            try:
                world.apply_settings(original_settings)
                now=world.get_settings()
                report['cleanup']['world_settings_restored']=(
                    now.synchronous_mode==original_settings.synchronous_mode and
                    now.fixed_delta_seconds==original_settings.fixed_delta_seconds and
                    now.no_rendering_mode==original_settings.no_rendering_mode)
            except Exception as e:report['cleanup']['world_settings_restore_error']=str(e)
        else:report['cleanup']['world_settings_restored']=True
        if report['status']=='completed' and not all(bool(v) for k,v in report['cleanup'].items() if k.endswith('restored') or k.endswith('destroyed') or k=='encoder_closed'):
            report['status']='cleanup_unverified'
        out.mkdir(parents=True,exist_ok=True)
        (out/'showcase_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print('SHOWCASE',report['status'],'video:',out/'showcase.mp4','report:',out/'showcase_report.json')
    return 0 if report['status']=='completed' else 2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true',help='Offline: inspect required source CSVs/JSON; never connect to CARLA')
    modes.add_argument('--inspect-server',action='store_true',help='Read-only CARLA actor/map/synchronization preflight')
    modes.add_argument('--record-showcase',action='store_true',help='Synchronous kinematic visual replay + automatic spectator + MP4 encoding')
    p.add_argument('--confirm-visual-replay',action='store_true',help='Acknowledges world tick ownership, temporary actors, spectator movement')
    p.add_argument('--project-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--scene',choices=d.SCENE_ORDER,action='append',help='Select one or more scenes in presentation order; default all')
    p.add_argument('--skip-missing-scenes',action='store_true',help='Explicitly permit missing archive scenes; do not fabricate data')
    p.add_argument('--slowdown',type=int,choices=[1,2,3,4],default=2,
                   help='Video + spectator slowdown (2=half speed, without changing sim dt)')
    p.add_argument('--width',type=int,default=960)
    p.add_argument('--height',type=int,default=540)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=20.)
    p.add_argument('--no-realtime',dest='realtime',action='store_false',default=True,
                   help='Render video with slowdown but do not pace CARLA spectator wall-clock playback')
    a=p.parse_args(argv)
    if a.width<640 or a.width>1920 or a.height<360 or a.height>1080 or a.width%2 or a.height%2:
        p.error('Video dimensions must be even: width 640..1920, height 360..1080')
    if a.confirm_visual_replay and not a.record_showcase:
        p.error('--confirm-visual-replay is valid only with --record-showcase')
    if a.record_showcase and not a.confirm_visual_replay:
        p.error('Recording requires --confirm-visual-replay')
    try:
        bundle=d.validate_sources(a.project_root,a.scene,skip_missing=a.skip_missing_scenes)
        print('Scenes:',', '.join(bundle['scenes']))
        for k,pth in bundle['paths'].items():
            print(('OK     ' if pth.is_file() else 'MISSING'),k,'=',pth)
        if bundle['missing']:
            print('Some scenes explicitly skipped because their archival source files are missing.')
        if a.check:
            print('OFFLINE CHECK PASSED. No CARLA connection or video recording occurred.')
            return 0
        import carla
        client=carla.Client(a.host,a.port);client.set_timeout(a.timeout)
        world=client.get_world()
        preflight=collision_free_preflight(carla,world,bundle)
        print('CARLA PREFLIGHT:',json.dumps(preflight,indent=2))
        if a.inspect_server: return 0
        return run(a,bundle,carla=carla)
    except (Exception,KeyboardInterrupt) as exc:
        print('SHOWCASE FAILED SAFE:',type(exc).__name__+':',str(exc))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
