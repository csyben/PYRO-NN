# Copyright [2019] [Christopher Syben, Markus Michen]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

def evaluation_filter(filter_one, filter_two, filter_three, output, dpi=600):
    fig = plt.figure(figsize=(20,10))

    zoomed_filter = plt.subplot2grid((1, 2), (0, 0))
    filter = plt.subplot2grid((1, 2), (0, 1))
    line_width = 5.0
    zoomed_filter.plot(np.arange(len(filter_one)),filter_one, linestyle='-.', color='#2ca02c',linewidth=line_width)
    zoomed_filter.plot(np.arange(len(filter_two)),filter_two,linestyle=':', color='#1f77b4',linewidth=line_width)
    zoomed_filter.plot(np.arange(len(filter_three)),filter_three,linestyle='--', color='#ff7f0e',linewidth=line_width)
    zoomed_filter.axes.set_xlim([0,len(filter_one)*0.005])
    zoomed_filter.axes.set_ylim([0, 0.005])
    zoomed_filter.tick_params(labelsize=24)
    zoomed_filter.legend(['Ramp', 'Ram-Lak', 'Learned'], loc='upper left',prop={'size': 28})

    shifted_filter_two = np.fft.ifftshift(filter_two, axes=0)
    shifted_filter_three = np.fft.ifftshift(filter_three, axes=0)

    interval = np.arange((-len(filter_one)//2)+1,(len(filter_one)//2)+1,1)
    filter.plot(interval, shifted_filter_two, color='#1f77b4',linewidth=line_width)
    filter.plot(interval, shifted_filter_three, linestyle='--', color='#ff7f0e',linewidth=line_width)
    filter.plot(interval, filter_two-filter_three, color='red',linewidth=line_width)
    filter.legend(['Ram-Lak', 'Learned', 'Diff.'], loc='upper center',prop={'size': 28})
    filter.tick_params(labelsize=24)
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, transparent=False, bbox_inches='tight')
    fig.show()

def evaluation_three(result_one, result_two, result_three, dim, output, dpi=600):
    result_one= np.squeeze(result_one)
    result_two = np.squeeze(result_two)
    result_three = np.squeeze(result_three)


    x0, y0 = 0, 0
    x1, y1 = dim[1]-1,dim[0]-1
    sample = 1000

    x,y = np.linspace(x0,x1,sample), np.linspace(y0,y1,sample)

    line_plots = []
    line_plots.append(scipy.ndimage.map_coordinates(result_one, np.vstack((x,y))))
    line_plots.append(scipy.ndimage.map_coordinates(result_two, np.vstack((x,y))))
    line_plots.append(scipy.ndimage.map_coordinates(result_three, np.vstack((x, y))))

    fig = plt.figure(figsize=(12,4))
    reco_1 = plt.subplot2grid((2, 3), (0, 0))
    reco_2 = plt.subplot2grid((2, 3), (0, 1))
    reco_3 = plt.subplot2grid((2, 3), (0, 2))

    line_plot1 = plt.subplot2grid((2, 3), (1, 0))
    line_plot2 = plt.subplot2grid((2, 3), (1, 1),sharey=line_plot1)
    line_plot3 = plt.subplot2grid((2, 3), (1, 2),sharey=line_plot1)


       # ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row')
    # -- Plot...
    #fig, axes = plt.subplots(nrows=2, ncols=3,sharex='col', sharey='row',figsize=(12,4))
    reco_1.imshow(result_one, cmap=plt.get_cmap('gist_gray'), vmin=-0.5, vmax=1.2)
    reco_1.plot([x0, x1], [y0, y1])
    reco_1.axis('off')
    reco_1.set_title('Ramp-Reco',fontsize=18, y=1.08)

    line_plot1.plot(line_plots[0])
    line_plot1.set_ylim([-0.5, 1.05])

    reco_2.imshow(result_two, cmap=plt.get_cmap('gist_gray'), vmin=-0.5, vmax=1.2)
    reco_2.plot([x0, x1], [y0, y1])
    reco_2.axis('off')
    reco_2.set_title('Ram-Lak-Reco',fontsize=18, y=1.08)

    line_plot2.plot(line_plots[1])
    line_plot2.set_ylim([-0.5, 1.05])

    reco_3.imshow(result_three, cmap=plt.get_cmap('gist_gray'), vmin=-0.5, vmax=1.2)
    reco_3.plot([x0, x1], [y0, y1])
    reco_3.axis('off')
    reco_3.set_title('Learned-Reco',fontsize=18, y=1.08)

    line_plot3.plot(line_plots[2])
    line_plot3.set_ylim([-0.5, 1.05])


    plt.tight_layout()

    plt.savefig(output, dpi=dpi,transparent=False,bbox_inches='tight')
    plt.show()
