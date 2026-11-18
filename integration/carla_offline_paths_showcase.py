"""Cinematic Town01 visualizer for the TWO ARCHIVED OFFLINE PLANNER PATHS.

No vehicle dynamics or planning is run here. Actors are physics-disabled and
teleported along verified archive CSV trajectories. An optional continuation
beyond the last recorded sample is ILLUSTRATIVE, not planner output. Requires
only the original stable observation, beam-200 report, and two plan sample CSVs.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import showcase_data_m6 as source

DT = .05
FPS = 20
DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path(__file__).resolve().parent / 'output/carla_showcase/offline_paths_extended_001'
# Desaturated, color-blind-friendly pale-blue and wheat. Not red/green.
COLORS = {
    'safety': (112, 157, 182),    # steel blue
    'corridor': (194, 163, 124),  # warm sandstone
    'extension': (147, 153, 163), # neutral gray
}
HEX = {k:'#%02x%02x%02x'%v for k,v in COLORS.items()}
SCENES=('overview','safety','corridor')


@dataclass(frozen=True)
class PlaybackPoint:
    x:float
    y:float
    yaw:float
    elapsed_s:float
    original:bool
    extension_m:float=0.0


def wrap_angle(theta):
    return math.atan2(math.sin(theta),math.cos(theta))


def lerp_angle(a,b,f):
    return wrap_angle(a+f*wrap_angle(b-a))


def sample_plan(rows, t):
    times=[source.number(r['t_s'],'t_s') for r in rows]
    if t <= times[0]:
        p=source.pose(rows[0],planning=True)
        return PlaybackPoint(p.x,p.y,p.yaw,0.,True)
    if t >= times[-1]-1e-10:
        p=source.pose(rows[-1],planning=True)
        return PlaybackPoint(p.x,p.y,p.yaw,times[-1],True)
    j=max(0,bisect.bisect_right(times,t)-1)
    a,b=source.pose(rows[j],planning=True),source.pose(rows[j+1],planning=True)
    f=(t-times[j])/(times[j+1]-times[j])
    return PlaybackPoint(a.x+(b.x-a.x)*f,a.y+(b.y-a.y)*f,
                         lerp_angle(a.yaw,b.yaw,f),t,True)


def make_playback(rows, extra_distance, extra_speed):
    if not (0<=extra_distance<=80 and .2<=extra_speed<=5.0):
        raise ValueError('Extra distance must be 0..80 m and illustrative speed must be 0.2..5 m/s')
    times=[source.number(r['t_s'],'t_s') for r in rows]
    if len(times)<2 or abs(times[0])>1e-7:
        raise ValueError('Planner CSV must begin at t=0 with at least two samples')
    if any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Planner sample times must be strictly increasing')
    if times[-1]>120:
        raise ValueError('Implausibly long source plan')
    n=max(1,math.ceil(times[-1]/DT))
    originals=[sample_plan(rows,min(times[-1],i*DT)) for i in range(n+1)]
    last=originals[-1]
    ext=[]
    for i in range(1,math.ceil(extra_distance/(extra_speed*DT)-1e-9)+1):
        dist=min(extra_distance,i*extra_speed*DT)
        ext.append(PlaybackPoint(last.x+dist*math.cos(last.yaw),
                     last.y+dist*math.sin(last.yaw),last.yaw,
                     last.elapsed_s+i*DT,False,dist))
    return originals+ext


def ghost_lines(playback):
    """Separate archived computed path from illustrative extension; no conflation."""
    archived=[p for p in playback if p.original]
    extra=([archived[-1]] if archived else [])+[p for p in playback if not p.original]
    return archived,extra


def check_bundle(project_root):
    bundle=source.validate_sources(project_root,('planning',))
    assert bundle['scenes']==['planning']
    return bundle


def to_carla_transform(carla,p,z,bb):
    """Place the visual BODY CENTER at the recorded body center; CARLA y is flipped."""
    theta=-p.yaw
    bx,by=float(bb.location.x),float(bb.location.y)
    x=p.x-bx*math.cos(theta)+by*math.sin(theta)
    y=-p.y-bx*math.sin(theta)-by*math.cos(theta)
    return carla.Transform(carla.Location(x=x,y=y,z=z),
                           carla.Rotation(yaw=-math.degrees(p.yaw)))


def camera_location(carla,target, road_z, style='follow'):
    theta=-target.yaw
    if style=='overview':
        back,height,side,pitch=7.,38.,0.,-80.
    else:
        back,height,side,pitch=10.,9.5,1.8,-33.
    return carla.Transform(carla.Location(x=target.x-back*math.cos(theta)-side*math.sin(theta),
                            y=-target.y-back*math.sin(theta)+side*math.cos(theta),
                            z=road_z+height),carla.Rotation(yaw=math.degrees(theta),pitch=pitch))


def preflight(world):
    if world.get_map().name.split('/')[-1]!='Town01':
        raise RuntimeError('A Town01 CARLA world is required')
    if world.get_settings().synchronous_mode:
        raise RuntimeError('World already synchronous; another client may own simulation ticks')
    actors=world.get_actors()
    occupied=[a for a in actors if a.type_id.startswith(('vehicle.','walker.','sensor.'))]
    if occupied:
        raise RuntimeError(f'World not isolated: {len(occupied)} vehicles/walkers/sensors detected')
    props=[a for a in actors if a.type_id.startswith('static.prop.')]
    return {'world_map':world.get_map().name,'other_vehicles_walkers_sensors':0,
            'existing_static_props':len(props),'world_synchronous_before':False}


def get_z(world,carla,p,backup=None):
    try:
        wp=world.get_map().get_waypoint(carla.Location(x=p.x,y=-p.y,z=1.0))
        if wp is not None:return float(wp.transform.location.z)
    except Exception:pass
    if backup is not None:return backup
    raise RuntimeError('Could not determine road Z for demonstration start')


def draw_segments(world,carla,points,colour,*,life_time,dotted=False,thickness=.055):
    pts=points
    for i,(a,b) in enumerate(zip(pts[:-1],pts[1:])):
        if dotted and i%4>=2:continue
        if math.hypot(a.x-b.x,a.y-b.y)<1e-5:continue
        z=get_z(world,carla,a,0.)+.25
        world.debug.draw_line(carla.Location(x=a.x,y=-a.y,z=z),
            carla.Location(x=b.x,y=-b.y,z=z),thickness=thickness,
            color=carla.Color(*colour),life_time=life_time,persistent_lines=False)


def get_font(size,bold=False):
    from PIL import ImageFont
    fonts=([r'C:\Windows\Fonts\segoeuib.ttf',r'C:\Windows\Fonts\arialbd.ttf'] if bold else
           [r'C:\Windows\Fonts\segoeui.ttf',r'C:\Windows\Fonts\arial.ttf'])
    fonts+=['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold
            else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for p in fonts:
        try:return ImageFont.truetype(p,size)
        except OSError:pass
    return ImageFont.load_default()


def camera_rgb_bytes(camera_img):
    """Convert an unannotated CARLA BGRA camera frame into ffmpeg RGB bytes."""
    from PIL import Image
    return Image.frombytes('RGB', (camera_img.width, camera_img.height),
                           bytes(camera_img.raw_data), 'raw', 'BGRX').tobytes()


def add_captions(camera_img,scene,index,total,frame,playback,cone_xy,src_names,report,slowed):
    from PIL import Image,ImageDraw
    canvas=Image.frombytes('RGBA',(camera_img.width,camera_img.height),
              bytes(camera_img.raw_data),'raw','BGRA')
    W,H=canvas.size
    overlay=Image.new('RGBA',canvas.size,(0,0,0,0));g=ImageDraw.Draw(overlay)
    font_head=get_font(max(19,int(W*.027)),True)
    font_body=get_font(max(13,int(W*.016)))
    font_sm=get_font(max(11,int(W*.012)))
    label={'overview':'The two planned trajectories',
           'safety':'Safety-first | clearance priority',
           'corridor':'Corridor-first | centerline priority'}[scene]
    g.rounded_rectangle((15,15,W-15,103),radius=12,fill=(15,24,35,228))
    g.text((32,23),f'{index}/{total}   {label}',font=font_head,fill=(242,244,246))
    g.text((32,71),'OFFLINE planned paths  •  no CARLA driving physics',font=font_body,fill=(205,214,225))

    # Small map in bottom-right shows the full source and any illustrative extension.
    x0,y0,x1,y1=W-295,H-257,W-17,H-102
    g.rounded_rectangle((x0,y0,x1,y1),radius=12,fill=(15,24,35,225))
    g.text((x0+10,y0+7),'Town01 path overview (CommonRoad XY)',font=font_sm,fill=(231,236,243))
    allp=[p for mode in ('safety','corridor') for p in playback[mode]]
    xvals=[p.x for p in allp]+[cone_xy[0]];yvals=[p.y for p in allp]+[cone_xy[1]]
    xmin,xmax=min(xvals),max(xvals);ymin,ymax=min(yvals),max(yvals)
    padx=max(2.5,(xmax-xmin)*.13);pady=max(3.,(ymax-ymin)*.25)
    xmin-=padx;xmax+=padx;ymin-=pady;ymax+=pady
    def xy(p):
        return (x0+14+(p.x-xmin)/(xmax-xmin)*(x1-x0-28),
                y1-20-(p.y-ymin)/(ymax-ymin)*(y1-y0-54))
    for mode in ('safety','corridor'):
        arch,extra=ghost_lines(playback[mode])
        if len(arch)>=2:g.line([xy(p) for p in arch],fill=(*COLORS[mode],255),width=3)
        for j in range(0,len(extra)-1,4):
            b=min(j+2,len(extra)-1)
            if b>j:g.line([xy(extra[j]),xy(extra[b])],fill=(*COLORS['extension'],240),width=2)
    c=xy(source.Pose(cone_xy[0],cone_xy[1],0.))
    g.ellipse((c[0]-4,c[1]-4,c[0]+4,c[1]+4),fill=(227,211,166,255))
    if scene!='overview':
        curr=frame
        sx,sy=xy(curr)
        g.ellipse((sx-5,sy-5,sx+5,sy+5),fill=(*COLORS[scene],255),outline=(255,255,255,255),width=1)

    g.rounded_rectangle((15,H-93,W-15,H-16),radius=10,fill=(15,24,35,232))
    if scene=='overview':
        g.text((29,H-84),'Steel blue = safety-first    |    Sand = corridor-first    |    Gray dashed = illustration',
               font=font_body,fill=(240,241,242))
        g.text((29,H-51),'Planning sources: '+src_names[0],font=font_sm,fill=(181,192,204))
        g.text((29,H-33),'Difference: %.2f m max time-aligned separation'%report['divergence']['max_time_aligned_separation_m'],
               font=font_sm,fill=(181,192,204))
    else:
        status=('ARCHIVED PLAN  —  original %0.1fs trajectory'%(frame.elapsed_s) if frame.original else
                'ILLUSTRATIVE EXTENSION  —  +%.1f m after archived endpoint'%frame.extension_m)
        g.text((29,H-84),status,font=font_body,
               fill=(210,229,237) if frame.original else (244,203,162))
        g.text((29,H-51),f'Source: {src_names[0 if scene=="safety" else 1]}',font=font_sm,fill=(181,192,204))
        g.text((29,H-33),'Physics disabled  •  spectator auto-follow  •  %dx slower presentation'%slowed,
               font=font_sm,fill=(181,192,204))
    return Image.alpha_composite(canvas,overlay).convert('RGB').tobytes()


class Film:
    def __init__(self,path,width,height,factor):
        import imageio_ffmpeg
        self.factor=factor;self.frames=0
        self.writer=imageio_ffmpeg.write_frames(str(path),(width,height),fps=FPS,
            codec='libx264',pix_fmt_out='yuv420p',output_params=['-movflags','+faststart'],
            ffmpeg_log_level='warning')
        self.writer.send(None)
    def send(self,frame):
        for _ in range(self.factor):
            self.writer.send(frame)
            self.frames+=1
    def close(self):
        if self.writer is not None:self.writer.close();self.writer=None


def make_sequence(bundle,args):
    s=bundle['data']
    return {mode:make_playback(s['plan_'+mode],args.extra_distance_m,args.extension_speed_mps)
            for mode in ('safety','corridor')}


def run_showcase(args,bundle,carla):
    client=carla.Client(args.host,args.port);client.set_timeout(args.timeout)
    if client.get_client_version().strip()!='0.9.16' or client.get_server_version().strip()!='0.9.16':
        raise RuntimeError('This CARLA showcase targets version 0.9.16')
    world=client.get_world();gate=preflight(world)
    out=args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory contains files; select a new --out')
    out.mkdir(parents=True,exist_ok=True)
    plans=make_sequence(bundle,args)
    observation=source.observation_pose(bundle['data']['observation'])
    ob_xy=[float(v) for v in bundle['data']['observation']['obstacle']['center_cr_xy_m']]
    src_names=[str(bundle['paths'][k].relative_to(bundle['root'])) for k in ('plan_safety','plan_corridor')]
    film=None;car=None;cone=None;rgb=None;change=False;events=[]
    q=queue.Queue()
    spectator=world.get_spectator()
    original_spectator=spectator.get_transform()
    old_settings=world.get_settings()
    details={'status':'aborted','mode':'offline_projected_paths_visual_reconstruction',
        'CARLA_driving_physics_enabled':False,'actual_rulebook_executed':False,
        'source_hashes':bundle['source_hashes'],
        'source_paths':{k:str(v) for k,v in bundle['paths'].items()},
        'planning_report_divergence':bundle['data']['plan_report']['divergence'],
        'extra_distance_m_per_path':args.extra_distance_m,
        'extension_speed_mps_illustrative':args.extension_speed_mps,
        'extension_is_planner_generated':False,
        'extension_road_or_obstacle_safety_verified':False,
        'slowdown_factor':args.slowdown,'fps':FPS,'camera_mode':'automatic follow / overhead',
        'data_overlay_enabled':not args.no_overlay, 'world_path_lines_enabled':True,
        'world_preflight':gate,'scenes':list(args.scenes),'scene_results':events,'cleanup':{},
        'note':'The original eight-second offline plans are reconstructed from CSV. Motion beyond their final archived samples is an openly labeled straight-tangent illustration. No autonomous planning, throttle, or brake is executed.'}
    try:
        from PIL import Image
        import imageio_ffmpeg
        new=world.get_settings();new.synchronous_mode=True;new.fixed_delta_seconds=DT;new.no_rendering_mode=False
        change=True;world.apply_settings(new)
        road_z=get_z(world,carla,observation)
        bp=world.get_blueprint_library().find('vehicle.tesla.model3')
        if bp.has_attribute('role_name'):bp.set_attribute('role_name','offline_showcase_only')
        car=world.try_spawn_actor(bp,carla.Transform(carla.Location(x=observation.x,y=-observation.y,z=road_z+.25),
                carla.Rotation(yaw=-math.degrees(observation.yaw))))
        if car is None:raise RuntimeError('Unable to spawn physics-disabled showcase vehicle')
        car.set_autopilot(False);car.set_simulate_physics(False)
        car.set_transform(to_carla_transform(carla,observation,road_z+.25,car.bounding_box))
        prop=world.get_blueprint_library().find('static.prop.constructioncone')
        cone_point=source.Pose(ob_xy[0],ob_xy[1],0.)
        cone_z=get_z(world,carla,cone_point,road_z)+.04
        cone=world.try_spawn_actor(prop,carla.Transform(carla.Location(x=cone_point.x,y=-cone_point.y,z=cone_z),carla.Rotation()))
        if cone is None:raise RuntimeError('Unable to spawn visual cone')
        if hasattr(cone,'set_simulate_physics'):cone.set_simulate_physics(False)
        cone.set_transform(to_carla_transform(carla,cone_point,cone_z,cone.bounding_box))
        rgb_bp=world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x',str(args.width));rgb_bp.set_attribute('image_size_y',str(args.height))
        rgb_bp.set_attribute('fov','88');rgb_bp.set_attribute('sensor_tick','0.0')
        rgb=world.spawn_actor(rgb_bp,camera_location(carla,observation,road_z,'overview'))
        if rgb is None:raise RuntimeError('Unable to spawn RGB recording camera')
        rgb.listen(q.put)
        film=Film(out/'offline_planned_paths.mp4',args.width,args.height,args.slowdown)
        overview_n=round(args.overview_seconds/DT)
        frame_index=0
        for sidx,scene in enumerate(args.scenes,1):
            count=overview_n if scene=='overview' else len(plans[scene])
            if not count:continue
            print(f'[{sidx}/{len(args.scenes)}] {scene}: {count} simulation frames')
            life_time=count*DT+.2
            for mode in ('safety','corridor'):
                archived,extension=ghost_lines(plans[mode])
                draw_segments(world,carla,archived,COLORS[mode],life_time=life_time)
                if args.extra_distance_m>0:
                    draw_segments(world,carla,extension,COLORS['extension'],life_time=life_time,dotted=True,thickness=.035)
            for i in range(count):
                begin=time.monotonic()
                p=(observation if scene=='overview' else plans[scene][i])
                frame=source.Pose(p.x,p.y,p.yaw) if scene!='overview' else observation
                z=get_z(world,carla,frame,road_z)+.25
                car.set_transform(to_carla_transform(carla,frame,z,car.bounding_box))
                if scene=='overview':
                    cam_mid=source.Pose((observation.x+ob_xy[0])*.5,
                        (observation.y+ob_xy[1])*.5,observation.yaw)
                    target=camera_location(carla,cam_mid,road_z,'overview')
                else:target=camera_location(carla,frame,z-.25,'follow')
                rgb.set_transform(target);spectator.set_transform(target)
                tick_id=int(world.tick(args.timeout))
                while True:
                    picture=q.get(timeout=args.timeout)
                    if int(picture.frame)>=tick_id:break
                if int(picture.frame)!=tick_id:
                    raise RuntimeError(f'RGB camera and world frames differ: {picture.frame} vs {tick_id}')
                if args.no_overlay:
                    data=camera_rgb_bytes(picture)
                else:
                    data=add_captions(picture,scene,sidx,len(args.scenes),
                        p if scene!='overview' else None,plans,ob_xy,src_names,
                        bundle['data']['plan_report'],args.slowdown)
                film.send(data)
                frame_index+=1
                if args.realtime:
                    gap=DT*args.slowdown-(time.monotonic()-begin)
                    if gap>0:time.sleep(gap)
            events.append({'scene':scene,'simulation_frames':count,'video_frames':count*args.slowdown,
                'estimated_video_duration_s':count*args.slowdown/FPS,
                'archived_duration_s':float(bundle['data']['plan_report']['settings']['horizon_steps'])*float(bundle['data']['plan_report']['settings']['dt_s']) if scene!='overview' else None,
                'illustrative_extension_m':args.extra_distance_m if scene!='overview' else 0})
        details['status']='completed'
        details['total_video_frames']=film.frames
        details['video_duration_s']=film.frames/FPS
    except KeyboardInterrupt:
        details['abort_reason']='Interrupted'
    except Exception as exc:
        details['abort_reason']=type(exc).__name__+': '+str(exc)
    finally:
        if film is not None:
            try:film.close();details['cleanup']['video_closed']=True
            except Exception as exc:details['cleanup']['video_close_error']=str(exc)
        if rgb is not None:
            try:rgb.stop();details['cleanup']['camera_destroyed']=bool(rgb.destroy())
            except Exception as exc:details['cleanup']['camera_error']=str(exc)
        else:details['cleanup']['camera_destroyed']=True
        if cone is not None:
            try:details['cleanup']['cone_destroyed']=bool(cone.destroy())
            except Exception as exc:details['cleanup']['cone_error']=str(exc)
        else:details['cleanup']['cone_destroyed']=True
        if car is not None:
            try:details['cleanup']['car_destroyed']=bool(car.destroy())
            except Exception as exc:details['cleanup']['car_error']=str(exc)
        else:details['cleanup']['car_destroyed']=True
        try:spectator.set_transform(original_spectator);details['cleanup']['spectator_restored']=True
        except Exception as exc:details['cleanup']['spectator_restore_error']=str(exc)
        if change:
            try:
                world.apply_settings(old_settings)
                settings=world.get_settings()
                details['cleanup']['world_settings_restored']= (
                    settings.synchronous_mode==old_settings.synchronous_mode and
                    settings.fixed_delta_seconds==old_settings.fixed_delta_seconds and
                    settings.no_rendering_mode==old_settings.no_rendering_mode)
            except Exception as exc:details['cleanup']['world_settings_restore_error']=str(exc)
        else:details['cleanup']['world_settings_restored']=True
        if details['status']=='completed' and any(v is False for k,v in details['cleanup'].items()):
            details['status']='cleanup_unverified'
        (out/'showcase_report.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
        print('SHOWCASE:',details['status'],'|',out/'offline_planned_paths.mp4')
        print('REPORT:',out/'showcase_report.json')
    return 0 if details['status']=='completed' else 2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true',help='Verify archives and preview duration without CARLA')
    mode.add_argument('--inspect-server',action='store_true',help='Read-only Town01 preflight')
    mode.add_argument('--record',action='store_true',help='Record moving, physics-disabled showcase in Town01')
    p.add_argument('--confirm-visual-replay',action='store_true')
    p.add_argument('--project-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--scene',dest='scenes',choices=SCENES,action='append',default=None,
                   help='Select scenes in order; default overview, safety, corridor')
    p.add_argument('--extra-distance-m',type=float,default=14.,
                   help='Illustrative continuation beyond each archived plan (0..80 meters); default 14')
    p.add_argument('--extension-speed-mps',type=float,default=1.4,
                   help='Visual-only illustrative continuation speed; default 1.4 m/s')
    p.add_argument('--slowdown',type=int,choices=(1,2,3,4),default=2,
                   help='Video and live spectator slowed (2 means half-speed)')
    p.add_argument('--overview-seconds',type=float,default=3.)
    p.add_argument('--no-overlay',action='store_true',
                   help='Record raw CARLA camera frames without captions, inset map, statistics, or source text; colored world-space trajectory lines remain')
    p.add_argument('--width',type=int,default=960)
    p.add_argument('--height',type=int,default=540)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=20.)
    p.add_argument('--no-realtime',dest='realtime',action='store_false',default=True,
                   help='Encode at rendering speed instead of pacing spectator wall time')
    a=p.parse_args(argv)
    if not (.1<=a.overview_seconds<=12):p.error('--overview-seconds must be .1..12 seconds')
    if not (0<=a.extra_distance_m<=80 and .2<=a.extension_speed_mps<=5):
        p.error('extra distance 0..80 m, extension speed 0.2..5 m/s')
    if a.width%2 or a.height%2 or not 640<=a.width<=1920 or not 360<=a.height<=1080:
        p.error('Width/height must be even, 640..1920 by 360..1080')
    if a.confirm_visual_replay and not a.record:p.error('--confirm-visual-replay requires --record')
    if a.record and not a.confirm_visual_replay:p.error('Recording requires --confirm-visual-replay')
    a.scenes=a.scenes or list(SCENES)
    try:
        bundle=check_bundle(a.project_root)
        plans=make_sequence(bundle,a)
        print('ARCHIVED PLANS:', ', '.join(bundle['data']['plan_report']['profiles'].keys()))
        print('Color palette: safety steel-blue, corridor sandstone, illustrative extension gray-dashed')
        print('Video data overlay:', 'OFF (raw CARLA camera; world path lines remain)' if a.no_overlay else 'ON')
        for scene in a.scenes:
            if scene=='overview':duration=a.overview_seconds*float(a.slowdown)
            else:duration=len(plans[scene])*DT*float(a.slowdown)
            print(f'Preview: {scene}: {duration:.1f}s video')
        print('Maximum original time-aligned separation:',
            bundle['data']['plan_report']['divergence']['max_time_aligned_separation_m'],'m')
        for k in ('plan_safety','plan_corridor','plan_report','observation'):
            print('SOURCE:',k,'=',bundle['paths'][k])
        if a.check:
            print('SOURCE CHECK PASSED. CARLA was not accessed.')
            return 0
        import carla
        if a.inspect_server:
            client=carla.Client(a.host,a.port);client.set_timeout(a.timeout)
            print('SERVER PREFLIGHT:',json.dumps(preflight(client.get_world()),indent=2))
            return 0
        return run_showcase(a,bundle,carla)
    except (Exception,KeyboardInterrupt) as exc:
        print('OFFLINE SHOWCASE FAILED:',type(exc).__name__+':',exc)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
