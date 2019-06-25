import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

sample_num = 5

def connect_2D_line(inputs_use, sample_num):


  sample_joint = np.reshape( np.asarray(inputs_use[sample_num]), (16,2))
  plt.figure()

  for i in range(len(sample_joint)):
    x, y = sample_joint[i]
    plt.scatter(x, y)

  plt.gca().invert_yaxis()



def connect_3D_line(outputs_use, targets_use, sample_num):

  # start_points = np.array([6, 2, 1, 6, 3, 4, 6, 7, 8, 8, 13, 14, 8, 12, 11])  # start points
  # end_points = np.array([2, 1, 0, 3, 4, 5, 7, 8, 9, 13, 14, 15, 12, 11, 10])  # end points

  start_points = np.array([6, 5, 4, 0, 1, 2, 0, 7, 8, 11, 12,     8, 14, 15, 8, 9])  # start points
  end_points = np.array(  [5, 4, 0, 1, 2, 3, 7, 8, 11, 12, 13,   14, 15, 16, 9, 10])  # end points

  # prediction list
  x_coord_p, y_coord_p, z_coord_p = [], [], []
  x_coord_sub_p, y_coord_sub_p, z_coord_sub_p = [], [], []

  # label list
  x_coord_l, y_coord_l, z_coord_l = [], [], []
  x_coord_sub_l, y_coord_sub_l, z_coord_sub_l = [], [], []

  pred = np.reshape(outputs_use[sample_num], (17,3))
  lb =  np.reshape(targets_use[sample_num], (17,3))


  fig = plt.figure()

  ax_pred = fig.add_subplot(121 , projection='3d')
  ax_pred.title.set_text('Predictions')

  ax_lb = fig.add_subplot(122, projection='3d')
  ax_lb.title.set_text('Labels')

  for idx in range(len(pred)):

    x_coor_p, y_coor_p, z_coor_p = pred[idx]
    x_coord_p.append(x_coor_p)
    y_coord_p.append(y_coor_p)
    z_coord_p.append(z_coor_p)

    x_coor_l, y_coor_l, z_coor_l = lb[idx]
    x_coord_l.append(x_coor_l)
    y_coord_l.append(y_coor_l)
    z_coord_l.append(z_coor_l)

  for i in range(len(start_points)):

    x_coord_sub_p.append([x_coord_p[start_points[i]], x_coord_p[end_points[i]]])
    y_coord_sub_p.append([y_coord_p[start_points[i]], y_coord_p[end_points[i]]])
    z_coord_sub_p.append([z_coord_p[start_points[i]], z_coord_p[end_points[i]]])

    x_coord_sub_l.append([x_coord_l[start_points[i]], x_coord_l[end_points[i]]])
    y_coord_sub_l.append([y_coord_l[start_points[i]], y_coord_l[end_points[i]]])
    z_coord_sub_l.append([z_coord_l[start_points[i]], z_coord_l[end_points[i]]])



  for j in range(len(start_points)):
    ax_pred.plot(x_coord_sub_p[j], y_coord_sub_p[j], z_coord_sub_p[j])
    ax_lb.plot(x_coord_sub_l[j], y_coord_sub_l[j], z_coord_sub_l[j])




# connect_2D_line(inputs_use, sample_num )
# connect_3D_line(outputs_use, targets_use, sample_num)



########################################################################
  # start_points = np.array([6, 5, 4, 0, 1, 2, 0, 7, 8, 11, 12, 8, 14, 15, 8, 9])  # start points
  # end_points = np.array([5, 4, 0, 1, 2, 3, 7, 8, 11, 12, 13, 14, 15, 16, 9, 10])  # end points
  #
  # # 2d inputs list
  # x_coord, y_coord = [], []
  # x_coord_sub, y_coord_sub = [], []
  #
  # # prediction list
  # x_coord_p, y_coord_p, z_coord_p = [], [], []
  # x_coord_sub_p, y_coord_sub_p, z_coord_sub_p = [], [], []
  #
  # # label list
  # x_coord_l, y_coord_l, z_coord_l = [], [], []
  # x_coord_sub_l, y_coord_sub_l, z_coord_sub_l = [], [], []
  #
  # pred = np.reshape(outputs_use[0], (17, 3))
  # lb = np.reshape(targets_use[0], (17, 3))
  #
  # inp = np.reshape(inputs_use[0], (16, 2))
  # inp = np.vstack(([0, 0], inp))
  #
  # fig2 = plt.figure()
  # ax_inp = fig2.add_subplot(131)
  # ax_inp.title.set_text('2D inputs')
  #
  # fig = plt.figure()
  #
  # ax_pred = fig.add_subplot(132, projection='3d')
  # ax_pred.title.set_text('Predictions')
  #
  # ax_lb = fig.add_subplot(133, projection='3d')
  # ax_lb.title.set_text('Labels')
  #
  # for idx in range(len(pred)):
  #   x_coor, y_coor = inp[idx]
  #   x_coord.append(x_coor)
  #   y_coord.append(y_coor)
  #
  #   x_coor_p, y_coor_p, z_coor_p = pred[idx]
  #   x_coord_p.append(x_coor_p)
  #   y_coord_p.append(y_coor_p)
  #   z_coord_p.append(z_coor_p)
  #
  #   x_coor_l, y_coor_l, z_coor_l = lb[idx]
  #   x_coord_l.append(x_coor_l)
  #   y_coord_l.append(y_coor_l)
  #   z_coord_l.append(z_coor_l)
  #
  # for i in range(len(start_points)):
  #   x_coord_sub.append([x_coord[start_points[i]], x_coord[end_points[i]]])
  #   y_coord_sub.append([y_coord[start_points[i]], y_coord[end_points[i]]])
  #
  #   x_coord_sub_p.append([x_coord_p[start_points[i]], x_coord_p[end_points[i]]])
  #   y_coord_sub_p.append([y_coord_p[start_points[i]], y_coord_p[end_points[i]]])
  #   z_coord_sub_p.append([z_coord_p[start_points[i]], z_coord_p[end_points[i]]])
  #
  #   x_coord_sub_l.append([x_coord_l[start_points[i]], x_coord_l[end_points[i]]])
  #   y_coord_sub_l.append([y_coord_l[start_points[i]], y_coord_l[end_points[i]]])
  #   z_coord_sub_l.append([z_coord_l[start_points[i]], z_coord_l[end_points[i]]])
  #
  # for j in range(len(start_points)):
  #   ax_inp.plot(x_coord_sub[j], y_coord_sub[j])
  #   ax_pred.plot(x_coord_sub_p[j], y_coord_sub_p[j], z_coord_sub_p[j])
  #   ax_lb.plot(x_coord_sub_l[j], y_coord_sub_l[j], z_coord_sub_l[j])
