"""
  The following scripts are adopted from SORT by Alex Bewley, which can be found at https://github.com/abewley/sort/blob/master/sort.py.  
"""
from __future__ import print_function
import numpy as np
from filterpy.kalman import KalmanFilter
from .association import giou_batch, hmiou, fish_iou_batch, ciou_batch

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
  
# Added
def bbox_center(bbox):
    """
    takes a bounding box and returns the center point of the box as [x_center, y_center]
    """
    return np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])

# Added
def areas(bboxs):
    """
    calculates the area of bounding boxes 
    """
    w = np.maximum(0, bboxs[:, 2] - bboxs[:, 0])
    h = np.maximum(0, bboxs[:, 3] - bboxs[:, 1])
    return w * h

# Added
def centers(bboxs):
    """
    calculates the center points of bounding boxes 
    """
    xc = (bboxs[:, 0] + bboxs[:, 2]) / 2
    yc = (bboxs[:, 1] + bboxs[:, 3]) / 2
    return np.stack((xc, yc), axis=1)

# Added
def normalized_distance(dirs_track):
  norm = np.linalg.norm(dirs_track, axis=1, keepdims=True)
  valid = norm[:, 0] > 1e-6
  dirs = np.zeros(dirs_track.shape)
  dirs[valid] = dirs_track[valid] / norm[valid]
  return dirs, valid

# Added
def fish_sim(det, track, track_dir, track_ages, IoUW, areaW, slipW, slip_decay):
  """
  Similarity score as described in work 
  """
  detbbox = det[:, :4]
  trackbbox = track[:, :4]

  # IoU
  iou= iou_batch(detbbox, trackbbox)

  # Distance
  det_area = areas(detbbox)[:, None]
  track_area = areas(trackbbox)[None, :]
  area_ratio = np.minimum(det_area, track_area) / (np.maximum(det_area, track_area) + 1e-6) # epsilin for numerical stability

  # Side-slip
  det_center = centers(detbbox)
  track_center = centers(trackbbox)

  dirs, valid = normalized_distance(track_dir)
  perpendicular_dirs = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)
  
  delta = det_center[:, None, :] - track_center[None, :, :]
  lateral = np.abs(np.sum(delta * perpendicular_dirs[ None, :, :], axis=2))

  track_area = np.sqrt(areas(trackbbox))[None, :]
  # epsilon for numerical stability
  slip = np.exp(-lateral / (2*track_area+1e-6))
  slip[: , ~valid] = 1.0

  if track_ages is None:
    slip_co = np.full((1, len(track)), slipW)
  else:
    #decay for older tracks
    tsu = np.asanyarray(track_ages)[None, :]
    slip_co = slipW * np.exp(-slip_decay * np.maximum(tsu -1, 0))

  IoUW_co = np.full((1, len(track)), IoUW)
  areaW_co = np.full((1, len(track)), areaW)
  
  weight_sum = IoUW_co + areaW_co + slip_co
  similarity = ((IoUW_co * iou) + (areaW_co * area_ratio) + (slip * slip_co)) / (weight_sum+1e-6)
  return np.clip(similarity, 0.0, 1.0)

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    
    self.obs_center = [bbox_center(bbox[:4])]

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

    self.obs_center.append(bbox_center(bbox[:4]))
    if len(self.obs_center)>5:
        self.obs_center = self.obs_center[-5:]


  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)
  
  #Added
  def get_dir(self):
    """
    Returns the current direction estimate.
    """
    if len(self.obs_center) < 2:
        return np.array([0.0, 0.0], dtype=float)
    diffs = np.diff(np.asarray(self.obs_center), axis=0)
    return np.mean(diffs, axis=0)



def associate_detections_to_trackers(detections,trackers, track_dirs, track_ages, iou_h, iou_l, decay_rate, IoUW=0.5, areaW=0.25, slipW=0.20, slip_decay=0.1, sim = "fishsim"):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  
  Modified to include decay of IOU threshold over time and choose similarity metric. 
  """
  
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  
  #Calculate decayed IOU threshold for each tracker
  track_ages = np.asarray(track_ages)
  track_thresh = np.maximum(iou_l, iou_h - decay_rate * track_ages)

  # Choose IoU/sim score for ablation. Default fish_sim
  if sim == "fishsim":
    fish_matrix = fish_sim(detections, trackers, track_dirs, track_ages=track_ages, IoUW=IoUW, areaW=areaW, slipW=slipW, slip_decay=slip_decay)
  elif sim == "fishiou":
    fish_matrix = fish_iou_batch(detections, trackers)
  elif sim == "iou":
    fish_matrix = iou_batch(detections, trackers)
  elif sim == "giou":
    fish_matrix = giou_batch(detections, trackers)
  elif sim == "hmiou":
    fish_matrix = hmiou(detections, trackers)
  elif sim == "ciou":
    fish_matrix = ciou_batch(detections, trackers)

  if min(fish_matrix.shape) > 0:
    # det i is matched to track j if sim exceeds the decayed threshold for that track
    a = (fish_matrix >= track_thresh.reshape(1, -1))
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      cost_matrix = 1 -fish_matrix.copy()
      # prevent matching for invalid pairs
      invalid = (a == 0)
      cost_matrix[invalid] = 1e+5
      matched_indices = linear_assignment(cost_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(fish_matrix[m[0], m[1]]< track_thresh[m[1]]):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class OurSort(object):
  def __init__(self, max_age=30, min_hits=3, birth_iou_thresh=0.6, birth_min_conf=0.35, iou_h=0.3, iou_l = 0.1, decay_rate = 0.05, 
               IoUW=0.55, areaW=0.25, slipW=0.20, slip_decay=0.5, use_birth_suppression  = True, det_min_conf = 0.3, sim = "fishsim"):
    """
    Sets key parameters for our tracker
    """
    self.max_age = max_age
    self.min_hits = min_hits

    # BIrth suppression 
    self.birth_iou_thresh = birth_iou_thresh
    self.birth_min_conf = birth_min_conf
    self.use_birth_suppression = use_birth_suppression
    self.det_min_conf = det_min_conf

    #Decay thresh 
    self.iou_h=iou_h
    self.iou_l = iou_l
    self.decay_rate = decay_rate
    
    # sim for ablation
    self.sim = sim

    # Weights for FishSim 
    self.IoUW = IoUW
    self.areaW = areaW
    self.slipW = slipW
    self.slip_decay = slip_decay

    self.trackers = []
    self.frame_count = 0

    
  # Added 
  def _get_birth_anchor_boxes(self):
      """
     Collects the bboxes of all curretly relevant tracks
     Such bboxes serve as anchors to decide whether an unmatched detection is a plausible new birth 
     or should be suppressed as an ID switch on an existing track.
      """

      anchors = []
      for trk in self.trackers:
          if trk.time_since_update <= self.max_age:
              box = trk.get_state()[0]
              anchors.append(box)
      if len(anchors) == 0:
          return np.empty((0, 4), dtype=float)
      return np.asarray(anchors, dtype=float)

  # Added
  def _filter_birth_candidates(self, dets, unmatched_dets, anchor_boxes):
      """
      Birth suppression logic

      Does the following:
      - detection confidence >= birth_min_conf = rejection
      - suppress if it overlaps too much with an existing track anchor
      """

      if len(unmatched_dets) == 0:
          return np.empty((0,), dtype=int)

      # higher-confidence unmatched detections first
      scores = dets[unmatched_dets, 4]
      order = np.argsort(-scores)

      accepted = []

      for rank_idx in order:
          det_idx = int(unmatched_dets[rank_idx])
          det = dets[det_idx]

          #box and confidence of current unmatched detection
          box = det[:4]
          conf = float(det[4])

          # min conf condition
          if conf < self.birth_min_conf:
              continue

          # overlap check 
          if len(anchor_boxes) > 0:
              max_iou_anchor = iou_batch(box.reshape(1, 4), anchor_boxes).max()
              if max_iou_anchor >= self.birth_iou_thresh:
                  continue

          accepted.append(det_idx)

      return np.asarray(accepted, dtype=int)
  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1

    # filter dets below min conf
    dets = dets[dets[:, 4] > self.det_min_conf]

    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
        pos = self.trackers[t].predict()[0]
        trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
        if np.any(np.isnan(pos)):
            to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
        self.trackers.pop(t)
    track_ages = np.array([max(0, trk.time_since_update - 1) for trk in self.trackers], dtype=float)
    track_dirs = np.array([trk.get_dir() for trk in self.trackers], dtype=float)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, track_dirs, track_ages, self.iou_h, self.iou_l, self.decay_rate,IoUW=self.IoUW, areaW=self.areaW, slipW=self.slipW, slip_decay=self.slip_decay, sim=self.sim)

    # update matched trackers with assigned detections
    for m in matched:
        self.trackers[m[1]].update(dets[m[0], :])
        
    # check for ablation
    if self.use_birth_suppression:
      anchor_boxes = self._get_birth_anchor_boxes()
      # run birth supp
      birth_candidates = self._filter_birth_candidates(dets, unmatched_dets, anchor_boxes)
    else:
      birth_candidates = unmatched_dets

    # create and initialise new trackers for unmatched detections
    for i in birth_candidates:
        trk = KalmanBoxTracker(dets[i, :])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
