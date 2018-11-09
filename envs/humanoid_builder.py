import xml.etree.ElementTree as ET
import os.path as osp
import numpy as np
import time

"""

'When I wrote this code, only God and I understood what I was doing. Now, only God knows.'
- The Unknown Programmer

"In programming, a 'magic number' is a value that should be given a symbolic name, 
but was instead slipped into the code as a literal, usually in more than one place."
- StackOverflow

"""


def build_hand(larm_size, 
               hand_pos, 
               right=True, 
               hand_type=1, 
               width=1, 
               thickness=1, 
               length=1, 
               hand_quat='1 0 0 0'
               ):
    side = 'right_' if right else 'left_'

    if hand_type==1:
        hand_size = '%.3f'%(0.04 * thickness)
        hand_str = "<geom name='"+side+"hand' pos='"+hand_pos+"' size='"+hand_size+"' type='sphere'/>\n"
    elif hand_type==2:
        sign = 1 if right else -1

        # magic numbers
        hand_l = 0.025
        hand_w = 0.035
        thumb_l = 0.025
        thumb_w = 0.01

        # hand position
        delta = float(larm_size)/2 + hand_l * length/2
        hand_pos = np.array([float(x) for x in hand_pos.split(' ')]) + np.array([delta, sign * delta, delta])
        hand_pos = '%.3f %.3f %.3f'%(hand_pos[0], hand_pos[1], hand_pos[2])

        # positions
        finger_pos = '0 %.3f 0'%(sign * (10/9) * 2 * hand_l * length)
        thumb_pos = '-%.3f 0 0'%(((10/9) * hand_w * width) + thumb_l * length)

        # sizes
        hand_size = '%.3f %.3f %.3f'%(hand_w * width, hand_l * length, 0.005 * thickness)
        thumb_size = '%.3f %.3f %.3f'%(thumb_l * length, thumb_w * width, 0.005 * thickness)

        # joint locs
        wrist_joint_pos = '0 %.3f 0'%(-sign * delta)
        knuckle_joint_pos = '0 %.3f 0'%(-sign * hand_l * length)
        thumb_joint_pos = '%.3f 0 0'%(thumb_l * length)

        # ranges
        wrist_y_range = '-30 130'
        knuckle_range = '-90 10'# if right else '-10 90'
        thumb_range = '-90 0'

        # axes
        knuckle_axis = '1 0 0' if right else '-1 0 0'

        hand_str = "<body name='"+side+"hand' pos='"+hand_pos+"' quat='"+hand_quat+"'>\n\
            <joint axis='1 0 0' name='"+side+"wrist_x' pos='"+wrist_joint_pos+"' range='-80 80' type='hinge'/>\n\
            <joint axis='0 1 0' name='"+side+"wrist_y' pos='"+wrist_joint_pos+"' range='"+wrist_y_range+"' type='hinge'/>\n\
            <!--<joint axis='0 0 1' name='"+side+"wrist_z' pos='"+wrist_joint_pos+"' range='-80 80' type='hinge'/>-->\n\
            <geom name='"+side+"hand1' type='box' size='"+hand_size+"' />\n\
            <body name='"+side+"fingers' pos='"+finger_pos+"'>\n\
                <joint axis='"+knuckle_axis+"' name='"+side+"knuckles' pos='"+knuckle_joint_pos+"' range='"+knuckle_range+"' type='hinge'/>\n\
                <geom name='"+side+"fingers' type='box' size='"+hand_size+"' pos='0 0 0'/>\n\
            </body>\n\
            <body name='"+side+"thumb' pos='"+thumb_pos+"'>\n\
                <joint axis='0 1 0' name='"+side+"thumb_y' pos='"+thumb_joint_pos+"' range='"+thumb_range+"' type='hinge'/>\n\
                <geom name='"+side+"thumb1' type='box' size='"+thumb_size+"' pos='0 0 0'/>\n\
            </body>\n\
        </body>"
    return ET.fromstring(hand_str)


def build_arm(right=True, 
              arm_width=1, 
              arm_thickness=1, 
              arm_length=1,
              hand_type=1,
              hand_width=1,
              hand_thickness=1,
              hand_length=1,
              ):

    side = 'right_' if right else 'left_'
    camera_line = "            <camera pos='0 0 0'/>\n" if right else ""

    axis1 = '2 1 1' if right else '2 -1 1'
    axis2 = '0 -1 1' if right else '0 1 1'
    axis3 = '0 -1 1' if right else '0 -1 -1'
    ang_range = '-85 60' if right else '-60 85'

    sign = 1 if right else -1

    uarm_pos = '0 %.3f 0.06'%(-sign * 0.17 * arm_width)
    uarm_fromto = '0 0 0 %.3f %.3f -%.3f'%(.16 * arm_length, -sign * .16 * arm_length, .16 * arm_length)
    larm_pos = '%.3f %.3f -%.3f'%(.18 * arm_length, -sign * .18 * arm_length, .18 * arm_length)
    larm_fromto = '0.01 %.3f 0.01 %.3f %.3f %.3f'%(sign * 0.01, .17 * arm_length, sign * .17 * arm_length, .17 * arm_length)
    hand_pos = '%.3f %.3f %.3f'%(.18 * arm_length, sign * .18 * arm_length, .18 * arm_length)

    uarm_size = '%.3f %.3f'%(0.04 * arm_thickness, 0.16 * arm_thickness)
    larm_size = '%.3f'%(0.031 * arm_thickness)
    hand_size = '%.3f'%(0.04 * arm_thickness)


    arm_str= "<body name='"+side+"upper_arm' pos='"+uarm_pos+"'>\n\
        <joint armature='0.0068' axis='"+axis1+"' name='"+side+"shoulder1' pos='0 0 0' range='"+ang_range+"' stiffness='1' type='hinge'/>\n\
        <joint armature='0.0051' axis='"+axis2+"' name='"+side+"shoulder2' pos='0 0 0' range='"+ang_range+"' stiffness='1' type='hinge'/>\n\
        <joint armature='0.0051' axis='0 -1 0' name='"+side+"shoulder3' pos='0 0 0' range='0 120' stiffness='1' type='hinge'/>\n\
        <geom fromto='"+uarm_fromto+"' name='"+side+"uarm1' size='"+uarm_size+"' type='capsule'/>\n\
        <body name='"+side+"lower_arm' pos='"+larm_pos+"'>\n\
            <joint armature='0.0028' axis='"+axis3+"' name='"+side+"elbow' pos='0 0 0' range='-90 50' stiffness='0' type='hinge'/>\n\
            <geom fromto='"+larm_fromto+"' name='"+side+"larm' size='"+larm_size+"' type='capsule'/>\n"\
            +camera_line+"\
        </body>\n\
    </body>\n"

    #            <geom name='"+side+"hand' pos='"+hand_pos+"' size='"+hand_size+"' type='sphere'/>\n"\

    arm_root = ET.fromstring(arm_str)
    larm_root = arm_root.find('body')
    if hand_type==2:


        def d_cos(v1, v2):
            return np.dot(v1, v2) / np.sqrt( np.dot(v1, v1) * np.dot(v2, v2) )

        def q_mult(q1, q2):
            a, b, c, d = q1[0], q1[1], q1[2], q1[3]
            e, f, g, h = q2[0], q2[1], q2[2], q2[3]
            q3 = [
                a * e - b * f - c * g - d * h,
                a * f + b * e + c * h - d * g,
                a * g - b * h + c * e + d * f,
                a * h + b * g - c * f + d * e
                ]
            return np.array(q3) 

        larm_vec = [float(x) for x in larm_fromto.split(' ')]
        larm_vec = np.array([larm_vec[3] - larm_vec[0], larm_vec[4] - larm_vec[1], larm_vec[5] - larm_vec[2]])    # lower arm direction vector
        
        # first rotation: around hand-fixed x-axis (which presently aligns with global x-axis)
        hand_vec = np.array([0., sign, 0.])    # hand body-fixed direction pointing from wrist to finger tips
        cos = d_cos(hand_vec, larm_vec)
        angle1 = sign * np.arccos(cos)
        q1 = np.array([np.cos(angle1/2), np.sin(angle1/2), 0, 0])
        hand_quat = '%.3f %.3f 0 0'%(q1[0], q1[1])

        # second rotation: around hand-fixed z-axis (which is presently rotated from global z-axis)
        hand_vec_y = sign * np.array([0, np.cos(angle1), np.sin(angle1)])    # hand-fixed y-axis expressed wrt global frame basis
        angle2 = - np.arccos(d_cos(hand_vec_y, larm_vec))
        hand_vec_z = sign * np.array([0, 0, 1])     # hand-fixed z-axis expressed wrt hand frame basis (because quaternions)
        q2 = np.concatenate([[np.cos(angle2/2)], np.sin(angle2/2) * hand_vec_z])

        # composed rotation by quaternion multiplication
        q3 = q_mult(q1, q2)
        hand_quat = '%.3f %.3f %.3f %.3f'%(q3[0], q3[1], q3[2], q3[3])
    else:
        hand_quat = '1 0 0 0'

    hand_root = build_hand(larm_size, hand_pos, right, hand_type, hand_width, hand_thickness, hand_length, hand_quat)
    larm_root.append(hand_root)

    return arm_root


def build_foot(shin_size, 
               foot_pos, 
               right=True, 
               foot_type=1, 
               foot_width=1, 
               foot_thickness=1, 
               foot_length=1
               ):
    side = 'right_' if right else 'left_'

    if foot_type==1:
        foot_size  = '%.3f'%(0.075 * foot_thickness)
        foot_str="<body name='"+side+"foot' pos='"+foot_pos+"'>\n\
                    <geom name='"+side+"foot' pos='0 0 0.1' size='"+foot_size+"' type='sphere' user='0'/>\n\
                </body>\n"
    elif foot_type==2:
        foot_l = 0.1 * foot_length
        geom_pos = '%.3f 0 0'%(foot_l/4)
        foot_size  = '%.3f %.3f %.3f'%(foot_l, 0.06 * foot_width, 0.028 * foot_thickness)
        foot_str="<body name='"+side+"foot' pos='"+foot_pos+"'>\n\
                    <joint armature='0.01' axis='0 1 0' name='"+side+"foot_y' pos='0 0 0' range='-45 20' stiffness='20' type='hinge'/>\n\
                    <joint armature='0.01' axis='1 0 0' name='"+side+"foot_x' pos='0 0 0' range='-35 35' stiffness='20' type='hinge'/>\n\
                    <geom name='"+side+"foot' pos='"+geom_pos+"' size='"+foot_size+"' type='box' user='0'/>\n\
                </body>\n"

    return ET.fromstring(foot_str)


def build_leg(right=True, 
              leg_width=1, 
              leg_thickness=1, 
              leg_length=1,
              foot_type=1,
              foot_width=1,
              foot_thickness=1,
              foot_length=1
              ):
    side = 'right_' if right else 'left_'

    axis1 = '1 0 0' if right else '-1 0 0'
    axis2 = '0 0 1' if right else '0 0 -1'

    thigh_size = '%.3f'%(0.06 * leg_thickness)
    shin_size  = '%.3f'%(0.049 * leg_thickness)

    sign = 1 if right else -1

    thigh_pos = '0 %.3f -0.04'%( -sign * 0.1 * leg_width)
    thigh_fromto = '0 0 0 0 %.3f %.3f'%(sign * 0.01, -0.34 * leg_length)
    shin_pos = '0 %.3f %.3f'%(sign * 0.01, -0.403 * leg_length)
    shin_fromto = '0 0 0 0 0 %.3f'%(-0.3 * leg_length )
    foot_pos = '0 0 %.3f'%(-0.45 * leg_length)

    leg_str = "<body name='"+side+"thigh' pos='"+thigh_pos+"'>\n\
        <joint armature='0.01' axis='"+axis1+"' damping='5' name='"+side+"hip_x' pos='0 0 0' range='-45 5' stiffness='10' type='hinge'/>\n\
        <joint armature='0.01' axis='"+axis2+"' damping='5' name='"+side+"hip_z' pos='0 0 0' range='-85 35' stiffness='10' type='hinge'/>\n\
        <joint armature='0.0080' axis='0 1 0' damping='5' name='"+side+"hip_y' pos='0 0 0' range='-110 75' stiffness='20' type='hinge'/>\n\
        <geom fromto='"+thigh_fromto+"' name='"+side+"thigh1' size='"+thigh_size+"' type='capsule'/>\n\
        <body name='"+side+"shin' pos='"+shin_pos+"'>\n\
            <joint armature='0.0060' axis='0 -1 0' name='"+side+"knee' pos='0 0 .02' range='-160 -2' type='hinge'/>\n\
            <geom fromto='"+shin_fromto+"' name='"+side+"shin1' size='"+shin_size+"' type='capsule'/>\n\
        </body>\n\
    </body>\n"

    leg_root = ET.fromstring(leg_str)
    shin_root = leg_root.find('body')

    foot_root = build_foot(shin_size, foot_pos, right, foot_type, foot_width, foot_thickness, foot_length)
    shin_root.append(foot_root)

    return leg_root

def build_lwaist(lwaist_z=1, 
                 lwaist_thickness=1,
                 lwaist_width=1,
                 pelvis_z=1,
                 butt_thickness=1,
                 butt_width=1
                 ):

    lwaist_size = '%.3f'%(0.06 * lwaist_thickness)
    butt_size = '%.3f'%(0.09 * butt_thickness)

    lwaist_pos = '-.01 0 -%.3f'%(0.26*lwaist_z)
    lwaist_fromto = '0 %.3f 0 0 %.3f 0'%(-0.06*lwaist_width, 0.06*lwaist_width)
    pelvis_pos = '0 0 -%.3f'%(0.165*pelvis_z)
    butt_fromto = '-.02 %.3f 0 -.02 %.3f 0'%(-0.07*butt_width, 0.07*butt_width)
    ab_joint_pos = '0 0 0.065'


    lwaist_str = "<body name='lwaist' pos='"+lwaist_pos+"' quat='1.000 0 -0.002 0'>\n\
        <geom fromto='"+lwaist_fromto+"' name='lwaist' size='"+lwaist_size+"' type='capsule'/>\n\
        <joint armature='0.02' axis='0 0 1' damping='5' name='abdomen_z' pos='"+ab_joint_pos+"' range='-45 45' stiffness='20' type='hinge'/>\n\
        <joint armature='0.02' axis='0 1 0' damping='5' name='abdomen_y' pos='"+ab_joint_pos+"' range='-75 30' stiffness='10' type='hinge'/>\n\
        <body name='pelvis' pos='"+pelvis_pos+"' quat='1.000 0 -0.002 0'>\n\
            <joint armature='0.02' axis='1 0 0' damping='5' name='abdomen_x' pos='0 0 0.1' range='-35 35' stiffness='10' type='hinge'/>\n\
            <joint armature='0.02' axis='0 1 0' damping='5' name='abdomen_y2' pos='"+ab_joint_pos+"' range='-75 30' stiffness='20' type='hinge'/>\n\
            <geom fromto='"+butt_fromto+"' name='butt' size='"+butt_size+"' type='capsule'/>\n\
        </body>\n\
    </body>\n"

    return ET.fromstring(lwaist_str)


def build_bottom(lwaist_z=1, 
                 lwaist_thickness=1,
                 lwaist_width=1,
                 pelvis_z=1,
                 butt_thickness=1,
                 butt_width=1,
                 leg_width=1,
                 leg_thickness=1,
                 leg_length=1,
                 foot_type=1,
                 foot_width=1,
                 foot_thickness=1,
                 foot_length=1
                 ):

    lwaist_root = build_lwaist(lwaist_z, 
                               lwaist_thickness, 
                               lwaist_width,
                               pelvis_z,
                               butt_thickness,
                               butt_width)

    rleg_root = build_leg(True, leg_width, leg_thickness, leg_length, foot_type, foot_width, foot_thickness, foot_length)
    lleg_root = build_leg(False, leg_width, leg_thickness, leg_length, foot_type, foot_width, foot_thickness, foot_length)

    pelvis_root = lwaist_root.find('body')
    pelvis_root.append(rleg_root)
    pelvis_root.append(lleg_root)

    return lwaist_root


def build_arms(arm_width=1,
               arm_thickness=1,
               arm_length=1,
               hand_type=1,
               hand_width=1,
               hand_thickness=1,
               hand_length=1
               ):

    rarm_root = build_arm(True, arm_width, arm_thickness, arm_length, hand_type, hand_width, hand_thickness, hand_length)
    larm_root = build_arm(False, arm_width, arm_thickness, arm_length, hand_type, hand_width, hand_thickness, hand_length)
    return rarm_root, larm_root

def build_torso(span=1, 
                neck=1, 
                thickness=1, 
                gut=1, 
                fixed=False
                ):

    torso_fromto = '0 %.3f 0 0 %.3f 0'%(-0.07 * span, 0.07 * span)
    torso_size = '%.3f'%(0.07 * thickness)
    head_pos = '0 0 %.3f'%(0.19 * neck)
    head_size = '%.3f'%(0.09 * thickness)
    nose_pos = '%.3f 0 %.3f'%(0.09 * thickness, 0.19 * neck)
    nose_size = '%.3f'%(0.02 * thickness)
    uwaist_fromto = '-.01 -.06 -.12 -.01 .06 -.12'
    uwaist_size = '%.3f'%(0.06 * gut)

    free_joint_str = "    <joint armature='0' damping='0' limited='false' name='root' pos='0 0 0' stiffness='0' type='free'/>\n" if not(fixed) else ""
    #free_joint_str = ""

    torso_str = "<body name='torso' pos='0 0 1.4'>\n"+free_joint_str+"\
        <camera name='track' mode='trackcom' pos='0 -4 0' xyaxes='1 0 0 0 0 1'/>\n\
        <geom fromto='"+torso_fromto+"' name='torso1' size='"+torso_size+"' type='capsule'/>\n\
        <geom fromto='"+uwaist_fromto+"' name='uwaist' size='"+uwaist_size+"' type='capsule'/>\n\
    </body>\n"

    #    <geom name='head' pos='"+head_pos+"' size='"+head_size+"' type='sphere' user='258'/>\n\

    head_str = "<body name='head' pos='0 0 0'>\n\
        <joint armature='0.01' axis='0 1 0' name='neck_y' pos='0 0 0' range='-45 60' stiffness='1' type='hinge'/>\n\
        <joint armature='0.01' axis='0 0 1' name='neck_z' pos='0 0 0' range='-75 75' stiffness='1' type='hinge'/>\n\
        <joint armature='0.01' axis='1 0 0' name='neck_x' pos='0 0 0' range='-25 25' stiffness='1' type='hinge'/>\n\
        <geom name='head1' pos='"+head_pos+"' size='"+head_size+"' type='sphere' user='258'/>\n\
        <geom name='nose' pos='"+nose_pos+"' size='"+nose_size+"' type='sphere'/>\n\
    </body>\n"

    torso_root = ET.fromstring(torso_str)
    head_root = ET.fromstring(head_str)
    torso_root.append(head_root)
    return torso_root


def build_motors(shoulder_scale=1, 
                 elbow_scale=1, 
                 ab_scale=1,
                 hip_scale=1,
                 knee_scale=1,
                 neck_scale=1,
                 hand_scale=1,
                 foot_scale=1,
                 hand_type=1,
                 foot_type=1
                 ):

    sh_range = '-%.3f %.3f'%(0.7 * shoulder_scale, 0.7 * shoulder_scale)
    el_range = '-%.3f %.3f'%(0.6 * elbow_scale, 0.6 * elbow_scale)
    ab_range = '-%.3f %.3f'%(0.4 * ab_scale, 0.4 * ab_scale)
    hip_range = '-%.3f %.3f'%(0.4 * hip_scale, 0.4 * hip_scale)
    knee_range = '-%.3f %.3f'%(0.4 * knee_scale, 0.4 * knee_scale)
    neck_range = '-%.3f %.3f'%(0.4 * neck_scale, 0.4 * neck_scale)
    hand_range = '-%.3f %.3f'%(0.4 * hand_scale, 0.4 * hand_scale)
    foot_range = '-%.3f %.3f'%(0.4 * foot_scale, 0.4 * foot_scale)

    if hand_type==2:
        hand_motor_str = "\
        <motor gear='25' joint='right_wrist_x' name='right_wrist_x' ctrllimited='true' ctrlrange='"+hand_range+"'/>\n\
        <motor gear='25' joint='right_wrist_y' name='right_wrist_y' ctrllimited='true' ctrlrange='"+hand_range+"'/>\n\
        <motor gear='25' joint='right_knuckles' name='right_knuckles' ctrllimited='true' ctrlrange='"+hand_range+"'/>\n\
        <motor gear='25' joint='right_thumb_y' name='right_thumb_y' ctrllimited='true' ctrlrange='"+hand_range+"'/>\n\
        <motor gear='25' joint='left_wrist_x' name='left_wrist_x' ctrllimited='true' ctrlrange='"+hand_range+"'/>\n\
        <motor gear='25' joint='left_wrist_y' name='left_wrist_y' ctrllimited='true' ctrlrange='"+hand_range+"'/>\n\
        <motor gear='25' joint='left_knuckles' name='left_knuckles' ctrllimited='true' ctrlrange='"+hand_range+"'/>\n\
        <motor gear='25' joint='left_thumb_y' name='left_thumb_y' ctrllimited='true' ctrlrange='"+hand_range+"'/>\n\
        "
    else:
        hand_motor_str = ""

    if foot_type==2:
        foot_motor_str = "\
        <motor gear='25' joint='right_foot_y' name='right_foot_y' ctrllimited='true' ctrlrange='"+foot_range+"'/>\n\
        <motor gear='25' joint='right_foot_x' name='right_foot_x' ctrllimited='true' ctrlrange='"+foot_range+"'/>\n\
        <motor gear='25' joint='left_foot_y' name='left_foot_y' ctrllimited='true' ctrlrange='"+foot_range+"'/>\n\
        <motor gear='25' joint='left_foot_x' name='left_foot_x' ctrllimited='true' ctrlrange='"+foot_range+"'/>\n\
        "
    else:
        foot_motor_str = ""

    motor_str = "<actuator>\n\
        <motor gear='100' joint='neck_y' name='neck_y' ctrllimited='true' ctrlrange='"+neck_range+"'/>\n\
        <motor gear='100' joint='neck_z' name='neck_z' ctrllimited='true' ctrlrange='"+neck_range+"'/>\n\
        <motor gear='100' joint='neck_x' name='neck_x' ctrllimited='true' ctrlrange='"+neck_range+"'/>\n\
        <motor gear='100' joint='abdomen_y' name='abdomen_y' ctrllimited='true' ctrlrange='"+ab_range+"'/>\n\
        <motor gear='100' joint='abdomen_y2' name='abdomen_y2' ctrllimited='true' ctrlrange='"+ab_range+"'/>\n\
        <motor gear='100' joint='abdomen_z' name='abdomen_z' ctrllimited='true' ctrlrange='"+ab_range+"'/>\n\
        <motor gear='100' joint='abdomen_x' name='abdomen_x' ctrllimited='true' ctrlrange='"+ab_range+"'/>\n\
        <motor gear='100' joint='right_hip_x' name='right_hip_x' ctrllimited='true' ctrlrange='"+hip_range+"'/>\n\
        <motor gear='100' joint='right_hip_z' name='right_hip_z' ctrllimited='true' ctrlrange='"+hip_range+"'/>\n\
        <motor gear='300' joint='right_hip_y' name='right_hip_y' ctrllimited='true' ctrlrange='"+hip_range+"'/>\n\
        <motor gear='200' joint='right_knee' name='right_knee' ctrllimited='true' ctrlrange='"+knee_range+"'/>\n\
        <motor gear='100' joint='left_hip_x' name='left_hip_x' ctrllimited='true' ctrlrange='"+hip_range+"'/>\n\
        <motor gear='100' joint='left_hip_z' name='left_hip_z' ctrllimited='true' ctrlrange='"+hip_range+"'/>\n\
        <motor gear='300' joint='left_hip_y' name='left_hip_y' ctrllimited='true' ctrlrange='"+hip_range+"'/>\n\
        <motor gear='200' joint='left_knee' name='left_knee' ctrllimited='true' ctrlrange='"+knee_range+"'/>\n\
        <motor gear='25' joint='right_shoulder1' name='right_shoulder1' ctrllimited='true' ctrlrange='"+sh_range+"'/>\n\
        <motor gear='25' joint='right_shoulder2' name='right_shoulder2' ctrllimited='true' ctrlrange='"+sh_range+"'/>\n\
        <motor gear='25' joint='right_shoulder3' name='right_shoulder3' ctrllimited='true' ctrlrange='"+sh_range+"'/>\n\
        <motor gear='25' joint='right_elbow' name='right_elbow' ctrllimited='true' ctrlrange='"+el_range+"'/>\n\
        <motor gear='25' joint='left_shoulder1' name='left_shoulder1' ctrllimited='true' ctrlrange='"+sh_range+"'/>\n\
        <motor gear='25' joint='left_shoulder2' name='left_shoulder2' ctrllimited='true' ctrlrange='"+sh_range+"'/>\n\
        <motor gear='25' joint='left_shoulder3' name='left_shoulder3' ctrllimited='true' ctrlrange='"+sh_range+"'/>\n\
        <motor gear='25' joint='left_elbow' name='left_elbow' ctrllimited='true' ctrlrange='"+el_range+"'/>\n\
        "+hand_motor_str+foot_motor_str+"\
    </actuator>\n"

    return ET.fromstring(motor_str)

MODEL_DIR = osp.abspath(osp.dirname(__file__))

def humanoid_xml_builder(fixed=False,
                         span=1,
                         neck=1,
                         torso_thickness=1,
                         gut_thickness=1,
                         arm_thickness=1,
                         arm_length=1,
                         hand_type=1,
                         hand_width=1,
                         hand_thickness=1,
                         hand_length=1,
                         lwaist_z=1, 
                         lwaist_width=1,
                         pelvis_z=1,
                         butt_thickness=1,
                         butt_width=1,
                         leg_width=1,
                         leg_thickness=1,
                         leg_length=1,
                         foot_type=1,
                         foot_width=1,
                         foot_thickness=1,
                         foot_length=1,
                         shoulder_scale=1,
                         elbow_scale=1,
                         ab_scale=1,
                         hip_scale=1,
                         knee_scale=1,
                         neck_scale=1,
                         hand_scale=1,
                         foot_scale=1
                         ):

    actuators = build_motors(shoulder_scale, 
                             elbow_scale,
                             ab_scale,
                             hip_scale,
                             knee_scale,
                             neck_scale,
                             hand_scale,
                             foot_scale,
                             hand_type,
                             foot_type)

    torso = build_torso(span, neck, torso_thickness, gut_thickness, fixed)

    rarm_root, larm_root = build_arms(span,
                                      arm_thickness,
                                      arm_length,
                                      hand_type,
                                      hand_width,
                                      hand_thickness,
                                      hand_length)

    lwaist_root = build_bottom(lwaist_z, 
                               gut_thickness,
                               lwaist_width,
                               pelvis_z,
                               butt_thickness,
                               butt_width,
                               leg_width,
                               leg_thickness,
                               leg_length,
                               foot_type,
                               foot_width,
                               foot_thickness,
                               foot_length)

    xml_tree = ET.parse(osp.join(MODEL_DIR, 'humanoid_template.xml'))
    root = xml_tree.getroot() 
    root.append(actuators)
    worldbody = root.find('worldbody')
    worldbody.append(torso)
    torso.append(lwaist_root)
    torso.append(rarm_root)
    torso.append(larm_root)
    hash_string = ''.join(str(time.time()).split('.')) + str(np.random.randint(1e6))
    raw_fname = 'test' + hash_string + '.xml'
    fname = osp.join(MODEL_DIR, raw_fname)
    xml_tree.write(fname)
    return fname
