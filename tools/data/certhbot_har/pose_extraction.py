import os
import argparse
from alive_progress import alive_bar
import mmcv
from tools.data.skeleton import ntu_pose_extraction

def parse_args():
  parser = argparse.ArgumentParser(
      description='Generate Pose Annotation for certhbot-har dataset')
  parser.add_argument('videos_path', type=str, help='path of the videos folder')
  parser.add_argument('output_path', type=str, help='path of the pose annotations to be stored')
  parser.add_argument('--device', type=str, default='cuda:0')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()

  videos_path = args.videos_path
  output_path = args.output_path
  ntu_pose_extraction.args.device = args.device

  detector_model = ntu_pose_extraction.init_detector(ntu_pose_extraction.args.det_config, ntu_pose_extraction.args.det_checkpoint, ntu_pose_extraction.args.device)
  assert detector_model.CLASSES[0] == 'person', ('We require you to use a detector '
                                            'trained on COCO')
  pose_model = ntu_pose_extraction.init_pose_model(ntu_pose_extraction.args.pose_config, ntu_pose_extraction.args.pose_checkpoint,
                                ntu_pose_extraction.args.device)

  for root, dirs, files in os.walk(videos_path):
    if not dirs:
      label = root.split('/')[-1]
      print(f"Label = {label}, number of videos = {len(files)}")
      with alive_bar(len(files), title=f'{label} videos') as bar:
        for i, f in enumerate(files):
          init=True if i==0 else False          
          anno = ntu_pose_extraction.ntu_pose_extraction(vid=os.path.join(root, f), skip_postproc=True, label=label, detector_model=detector_model, pose_model=pose_model)
          mmcv.dump(anno, os.path.join(output_path, label, f.split('.')[0] + '.pkl'))
          bar()