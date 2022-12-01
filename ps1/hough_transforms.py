import cv2
import numpy as np


def hough_lines_acc(BW, RhoResolution=0.5, Theta=np.arange(-90, 89, 0.5)):
  theta_rad = Theta / 180.0 * np.pi
  y, x = np.where(BW==255)
  xy = np.vstack([x,y]).transpose()
  cossin = np.vstack([np.cos(theta_rad), np.sin(theta_rad)])
  rhos = np.matmul(xy, cossin)
  rho_min = int(np.floor(rhos.min() / RhoResolution))
  rho_max = int(np.ceil(rhos.max() / RhoResolution))
  rho_edges = RhoResolution * np.linspace(rho_min, rho_max, rho_max-rho_min+1)

  num_theta, num_rho = len(Theta), len(rho_edges) - 1
  
  H = np.zeros([num_rho, num_theta], dtype=int)
  for tt in np.arange(num_theta):
    h, e = np.histogram(rhos[:, tt], rho_edges)
    H[:, tt] = h

  rho = e[:-1] + (e[1] - e[0])/2
  
  return {'H':H, 'theta':Theta, 'rho':rho}

def hough_peaks(H, numpeaks):
  index = np.argsort(H, axis=None)
  keep_index = index[-numpeaks:]
  row_indices, col_indices = np.unravel_index(keep_index, H.shape)
  
  return np.vstack([row_indices, col_indices]).transpose()

def hough_peaks_highlight(H, peaks):
  H_gray = (H/H.max() * 255.0).astype(np.uint8)
  H_color = cv2.cvtColor(H_gray, cv2.COLOR_GRAY2BGR)
  for (row, col) in peaks:
    cv2.circle(H_color, (col, row), radius=4, color=(0,255,0))
  return H_color

def hough_lines_draw(img, ds, ts):
  """
  draw bunch of lines onto img
  ds: distance from point of origin to each line
  ts: angle from x axis to line's normal
  """
  if len(img.shape) == 2:
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  else:
    img_color = img

  height, width = img_color.shape[0], img_color.shape[1]
  x = np.array([0, width-1])
  y = np.array([0, height-1]) 
  
  for i in range(len(ds)):
    d, t = ds[i], ts[i]
    t = t / 180.0 * np.pi
    if t:
      y_ = ((d - x*np.cos(t))/np.sin(t)).astype(int)
      p1 = (x[0], y_[0])
      p2 = (x[1], y_[1])
    else:
      x_ = ((d - y*np.sin(t))/np.cos(t)).astype(int)
      p1 = (x_[0], y[0])
      p2 = (x_[1], y[1])
    cv2.line(img_color, p1, p2, (0,255,0), thickness=2)    
  return img_color

def hough_circles_acc(edges, thetas, radius):
  """
  hough transform using estimated gradient theta
  """
  H = np.zeros(edges.shape, dtype=int)
  y, x = np.where(edges==255)
  
  # thetas is gradient from black to white
  # so it can detect white coin on black ground
  # adding pi to detect also black coin on white ground
  ts = thetas[y,x]
  ts = np.array([ts, ts+np.pi])
  
  #white on black coins
  a = (x + radius * np.cos(ts)).astype(int).reshape(1,-1)
  b = (y + radius * np.sin(ts)).astype(int).reshape(1,-1)

  select = (a < H.shape[1]) & (b < H.shape[0])
  centers = np.vstack([b[select], a[select]])
  center_uniques, cnts = np.unique(centers, axis=1, return_counts=True)
  H[center_uniques[0], center_uniques[1]] = cnts

  return H

def hough_circles_draw(img, cs, rs):
  """
  draw circles on img
  cs: list of (y, x) centers
  rs: radius or list of radius for each center
  """
  if len(img.shape) == 2:
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  else:
    img_color = img
  
  if isinstance(rs, int):
    rs = [rs]*len(cs)

  for i in range(len(cs)):
    y, x = cs[i]
    r = rs[i]
    cv2.circle(img_color, (x, y), r, (0,255,0), thickness=2)
  
  return img_color

def hough_circles_acc_multi_radius(edges, thetas, radius_range=(20,50)):
  radius_num = radius_range[1] - radius_range[0] 
  assert radius_num > 0
  H = np.zeros([radius_num, edges.shape[0], edges.shape[1]])
  for ri in np.arange(radius_num):
    radius = ri + radius_range[0]
    H[ri] = hough_circles_acc(edges, thetas, radius)

  return H


def find_circles(edges, thetas, radius_range=(20,50), num_peaks=50):
  H = hough_circles_acc_multi_radius(edges, thetas, radius_range)
  
  R = np.argmax(H, axis=0)
  H = np.amax(H, axis=0)

  # threshold = H.max() / 2
  # H = (H > threshold) * H
  
  peaks = hough_peaks(H, num_peaks)
  y, x = peaks[:, 0], peaks[:, 1]
  r = radius_range[0] + R[y, x]
  
  C = np.vstack([y,x,r]).transpose()
  
  return C, H

def circle_nms(H, nms_threshold=5):
  """
  nms-like post processing, using euclidean distance instead of IoU
  to keep only the brightest spot
  """
  y, x = np.where(H>0)
  vals = H[y, x].transpose()
  centers = np.vstack([y,x,vals]).transpose()
  centers = centers[vals.argsort()]

  keeps = []

  while len(centers) > 0:
    last_center = centers[-1, :-1]
    dist = (centers[:, :-1] - last_center)**2
    dist = np.sqrt(np.sum(dist, axis=1))

    keeps.append(last_center)
    #delete every pixels in 5-pixel vicinity
    remain = dist > nms_threshold
    centers = centers[remain]

  return np.array(keeps, dtype=int)
  

def find_circles2(edges, thetas, vote_threshold=0.5, nms_threshold=5, radius_range=(20,50)):
  """
  using nms to eliminate local non-maxima instead of numpeaks,
  so only 1 circle is drawn for each coin
  """
  H = hough_circles_acc_multi_radius(edges, thetas, radius_range)
  
  R = np.argmax(H, axis=0)
  H = np.amax(H, axis=0)

  threshold = H.max() * vote_threshold
  H2 = (H > threshold) * H
  
  peaks = circle_nms(H2, nms_threshold)
  y, x = peaks[:, 0], peaks[:, 1]
  
  radius = R[y, x] + radius_range[0]

  C = np.vstack([y,x,radius]).transpose()
  
  return C, H

def refine_lines(edges, H, rho, theta, initial_peaks=100, parallel_relax=10, pen_width = 20, min_length=200):
  initial_peaks = 100
  index = H.argsort(axis=None)[-initial_peaks:]
  d_indices, t_indices = np.unravel_index(index, H.shape)
  ds = rho[d_indices]
  ts = theta[t_indices]
  votes = H.flatten()[index]
  lines = np.vstack([ds, ts, votes]).transpose()
  
  # step 1: remove paralel lines too close to each other
  # remain_lines = []
  lines_1 = []
  while len(lines) > 0:
    most_likely_line = lines[-1,:]
    lines_1.append(most_likely_line)
    diff = lines - most_likely_line
    distance_diff = np.abs(diff[:, 0])
    angle_diff = np.abs(diff[:, 1])
    parallel = (angle_diff <= parallel_relax)
    close = (distance_diff < pen_width*0.9)
    deletes = parallel & close
    lines = lines[~deletes]
    # remain_lines.append(lines)

  # return remain_lines  
  # return np.array(lines_1)

  # step 2: only keep lines that has parallel about pen_width apart
  lines_2 = []
  for line in lines_1:
    diff = lines_1 - line
    distance_diff = np.abs(diff[:, 0])
    angle_diff = np.abs(diff[:, 1])
    parallel = (angle_diff <= parallel_relax)
    close = (distance_diff >= pen_width*0.9) & (distance_diff < pen_width*1.2)
    if any(parallel & close):
      lines_2.append(line)

  # return np.array(lines_2)

  # step 3: check line length
  discontinuous_thres = 30**2
  lines_3 = []
  for i, (d, t, v) in enumerate(lines_2):
    black = np.zeros(edges.shape, dtype=np.uint8)
    line_img = hough_lines_draw(black, [d], [t])
    line_img = line_img[:,:,1] / 255.0
    overlay = line_img * edges
    
    y,x = np.where(overlay > 0)
    # sort the points along the line using x index - not the best but acceptable
    sort_idx = x.argsort()
    line_points = np.vstack([y[sort_idx],x[sort_idx]])
    # calculate distance from one point to its next neighbor
    dist = (np.diff(line_points, axis=1))**2
    dist = dist.sum(axis=0)
    #find the continuous segment using threshold
    has_next = (dist < discontinuous_thres).astype(np.int8)
    has_next = np.concatenate(([0],has_next,[0]))
    segments_edges = np.abs(np.diff(has_next))
    segments = np.where(segments_edges==1)[0].reshape(-1,2)
    # find the length of each segment
    segment_lens = []
    for si, (start, stop) in enumerate(segments):
      p1 = line_points[:, start]
      p2 = line_points[:, stop-1]
      segment_lens.append(((p2-p1)**2).sum())
    if max(segment_lens) > min_length**2:
      lines_3.append((d,t,v))

  return np.array(lines_3)

def refine_circles(edges, circles, overlay_thres=0.15, contain_thres=100):
  """
  remove cirles that either:
  - dont have enough edge point on it (define by overlay_thres ratio)
  - or has too much edge point inside of it (define by contain_thres)

  """
  circles_1 = []
  # remove case 1
  # print("pass 1")
  for (y, x, r) in circles:
    black = np.zeros(edges.shape, dtype=np.uint8)
    hollow_circle = cv2.circle(black, (x, y), r, (1), thickness=2)
    overlay = hollow_circle * edges
    circle_pts = np.sum(hollow_circle > 0)
    overlay_pts = np.sum(overlay > 0)
    ratio = overlay_pts/circle_pts
    if ratio > overlay_thres:
      circles_1.append((y,x,r))
    # print(f'{x},{y},{r}: overlay ratio={ratio}, overlay_thres={overlay_thres}, remove={ratio<=overlay_thres}')
    

  # remove case 2
  circles_2 = []
  # print('pass 2')
  for (y, x, r) in circles_1:
    black = np.zeros(edges.shape, dtype=np.uint8)
    filled_circle = cv2.circle(black, (x, y), r-1, (1), thickness=-1)
    overlay = filled_circle * edges
    contain = np.sum(overlay > 0)
    if contain < contain_thres:
      circles_2.append((y,x,r))
    # print(f'{x},{y},{r}: contain={contain}, contain_thres={contain_thres}, remove={contain>=contain_thres}')

  return np.array(circles_2)

  