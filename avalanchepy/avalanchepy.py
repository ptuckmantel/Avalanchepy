from skimage.filters import threshold_otsu as otsu
import numpy as np
import matplotlib.pyplot as plt

import scipy
from module import utils2

class Extract_events:
    def __init__(self, raw_imgs):
        self.raw_imgs = raw_imgs
        self.bin_imgs = None
        self.switchmap = None
        self.eventmap = None
        self.contours_ini = None
        self.event_contours = None
        self.events_ordered = None
        self.event_distrib = None



    def binarise_series(self, th='Auto'):
        tmp_bin = []
        first_img = self.raw_imgs[0]
        if th == 'Auto':
            thresh = otsu(first_img)
        else:
            thresh = th
        for i in self.raw_imgs:
            tmp_bin.append(i > thresh)
        self.bin_imgs = tmp_bin

    def invert_pha_bin(self):
        tmp = []
        for i in self.bin_imgs:
            tmp.append(np.logical_not(i))
        self.bin_imgs = tmp

    def get_switchmap(self, method='median'):
        """ Creates a switching map showing the switching scan as a function of position

        Keyword arguments:
        -----------------
        method: which method is used to determine when switching occurs (default 'median'). Median chooses the median switching event. Maximum chooses the last

        """

        pha_bin = []
        for i in range(len(self.bin_imgs)):
            pha_bin.append(self.bin_imgs[i].astype(float))
        test_tau_der = np.zeros_like(pha_bin[0].astype(float))

        for i in range(pha_bin[0].shape[0]):
            for j in range(pha_bin[0].shape[1]):
                tmp = []
                for k in pha_bin:
                    tmp.append(k[i, j])
                if tmp[0] == tmp[-1]:
                    test_tau_der[i, j] = np.nan

                else:
                    #tmp_forward = tmp[tmp > 0]
                    changes = utils2.wherechanged(tmp)
                    if utils2.checkequal(tmp) == True:
                        switch_scan = np.nan
                    else:
                        if method == 'maximum':
                            switch_scan = np.ceil(np.nanmax(changes))
                        elif method == 'median':
                            switch_scan = np.ceil(np.nanmedian(changes))
                    test_tau_der[i, j] = switch_scan

        self.switchmap = test_tau_der

    def get_jump_surfaces_by_type_switchmap(self, box_size=2, filter_size=1, struc_elem=True, plotmap=True,
                                            figname='totalmap', allplots=False):
        """ Extracts the map of nucleation, motion and merging and the corresponding jump surfaces in pixels using the switching map.

        Arguments:
        ---------
        switchmap: map of switching scans
        pha_bin: array of binarised phases of the successive PFM scans. Note: The as-grown state has to be 0
        cutoff_time: number of scan up to which the analysis is to be done

        Keyword arguments:
        -----------------
        box_size: range of newly switched pixels when looking for connected domains (default: 1)
        filter_size: minimum switching event size that is to be considered (default: 1)
        struc_elem: if True, diagonal pixels are considered as part of the same domain (default: True)
        plotmap: if True, plots and saves the event map with of switching events(default: True)
        allplots: if True, plots and saves all intermediary event maps with name figname_(scan number). (default True)
        figname: name of the plot file without extension. Format: png (default: totalmap)

        Returns:
        -------
        totalmap: Map of nucleation (1), motion (2) and merging (3) events
        jump_surfaces: array of successive jump surfaces. 0: total, 1: nucleation, 2: motion, 3: merging events 4) errors. Each of these is an array where each element is the list of all jumps for that particular scan number

        """
        switchmap = np.copy(self.switchmap)
        pha_bin = self.bin_imgs
        cmap = plt.get_cmap('brg', 3)
        cmap.set_over('gray')
        cont0 = utils2.get_contours_ini(pha_bin[0])
        contours = utils2.get_contours_full(switchmap, len(pha_bin))
        switchmap[np.isnan(switchmap)] = 999.
        # print switchmap

        structuring_element = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        totalmap = np.zeros_like(switchmap.astype(float))
        totalmap[:] = np.nan

        alljumps_tot = []
        alljumps_nucl = []
        alljumps_mot = []
        alljumps_merg = []
        alljumps_error = []

        for i in range(len(pha_bin)):
            # Extract where switching occured
            start = switchmap <= i - 1
            end = switchmap <= i
            diff = end.astype(int) - start.astype(int)

            alljumps_tot_1img = []
            alljumps_nucl_1img = []
            alljumps_mot_1img = []
            alljumps_merg_1img = []
            alljumps_error_1img = []

            # Label areas where switching occured
            if struc_elem == True:
                labeled_diff, num_features = scipy.ndimage.measurements.label(diff, structuring_element)
            elif struc_elem == False:
                labeled_diff, num_features = scipy.ndimage.measurements.label(diff)
            boxes_1scan = np.zeros_like(labeled_diff)

            for m in range(1, num_features + 1):
                # Get positions of all points of each label
                ind = np.nonzero(labeled_diff == m)
                # label_surface=len(ind)
                label_surface = np.sum(labeled_diff == m)
                box = np.zeros_like(labeled_diff)
                boxed_pha_bin = np.zeros_like(labeled_diff)
                # define box around each labelled pixel
                for j in range(len(ind[0])):
                    tmp_labels = []
                    x = ind[0][j]
                    y = ind[1][j]
                    # print x,y
                    if x <= box_size:
                        left = 0
                    else:
                        left = x - box_size
                    if x >= switchmap.shape[0] - box_size:
                        right = switchmap.shape[0] - 1
                    else:
                        right = x + box_size
                    if y <= box_size:
                        top = 0
                    else:
                        top = y - box_size
                    if y >= switchmap.shape[1] - box_size:
                        bottom = switchmap.shape[1] - 1
                    else:
                        bottom = y + box_size
                        # Get map of all points within box
                    for tmp_x in range(left, right + 1):
                        for tmp_y in range(top, bottom + 1):
                            box[tmp_x, tmp_y] = 1
                            boxes_1scan[tmp_x, tmp_y] = 1

                tmpstart = np.copy(start)
                tmpstart = np.nan_to_num(tmpstart)
                # Get switched areas up to the previous scan, that are in the box
                boxed_pha_bin = (tmpstart + pha_bin[0]) * box

                # Label the switched areas in the box
                if struc_elem == True:
                    labeled_boxed_pha_bin, num_connectors = scipy.ndimage.measurements.label(boxed_pha_bin,
                                                                                             structuring_element)
                elif struc_elem == False:
                    labeled_boxed_pha_bin, num_connectors = scipy.ndimage.measurements.label(boxed_pha_bin)
                    # Convert the labeled array into a 1d-array of (unique) label values whose length gives the number of domains connecting
                # the newly switched area
                num_pix = np.shape(pha_bin[0])[0] * np.shape(pha_bin[0])[1]
                boxed_labels = np.reshape(labeled_boxed_pha_bin, (1, num_pix))
                boxed_labels = np.unique(boxed_labels[boxed_labels != 0])
                number_of_connecting_doms = len(boxed_labels)

                # Define event type and append all points of that event to the totalmap
                # nucleation
                alljumps_tot_1img.append(label_surface)
                if number_of_connecting_doms == 0:
                    eventtype = 2
                    alljumps_nucl_1img.append(label_surface)
                    for j in range(len(ind[0])):
                        x = ind[0][j]
                        y = ind[1][j]
                        totalmap[x, y] = 2.

                # motion
                elif number_of_connecting_doms == 1:
                    eventtype = 1
                    alljumps_mot_1img.append(label_surface)
                    for j in range(len(ind[0])):
                        x = ind[0][j]
                        y = ind[1][j]
                        totalmap[x, y] = 1.
                # merging
                elif number_of_connecting_doms > 1:
                    eventtype = 3
                    alljumps_merg_1img.append(label_surface)
                    for j in range(len(ind[0])):
                        x = ind[0][j]
                        y = ind[1][j]
                        totalmap[x, y] = 3.
                # undefined
                else:
                    eventtype = 4
                    alljumps_error_1img.append(label_surface)
                    for j in range(len(ind[0])):
                        x = ind[0][j]
                        y = ind[1][j]
                        totalmap[x, y] = 4.

            if allplots == True:
                plt.imshow(totalmap, vmin=1, vmax=3, cmap=cmap)
                for cont in cont0:
                    plt.plot(cont[:, 1], cont[:, 0], color='cyan', lw=0.5)
                for n in range(i + 1):
                    for cont in contours[n]:
                        plt.plot(cont[:, 1], cont[:, 0], color='black', lw=0.3)
                # plt.colorbar()
                plt.savefig(figname + str(i) + '.png')
                plt.show()
                plt.close()
            alljumps_tot.append(alljumps_tot_1img)
            alljumps_nucl.append(alljumps_nucl_1img)
            alljumps_mot.append(alljumps_mot_1img)
            alljumps_merg.append(alljumps_merg_1img)
            alljumps_error.append(alljumps_error_1img)

        if plotmap == True:
            cont0 = utils2.get_contours_ini(pha_bin[0])
            contours = utils2.get_contours_full(switchmap, len(pha_bin))
            plt.imshow(totalmap, vmin=1, vmax=3, cmap=cmap)
            for cont in cont0:
                plt.plot(cont[:, 1], cont[:, 0], color='cyan', lw=0.5)
            for i in range(len(pha_bin)):
                for cont in contours[i]:
                    plt.plot(cont[:, 1], cont[:, 0], color='black', lw=0.3)
            # plt.colorbar()
            plt.axis('off')
            plt.savefig(figname + '.png', dpi=300)

            plt.show()

        cont0 = utils2.get_contours_ini(pha_bin[0])
        contours = utils2.get_contours_full(switchmap, len(pha_bin))
        self.events_ordered = (alljumps_tot, alljumps_nucl, alljumps_mot, alljumps_merg, alljumps_error)
        self.eventmap = totalmap
        self.contours_ini = cont0
        self.event_contours = contours

    def flatten_list(self):
        flattened_all = []
        for l in self.events_ordered:
            flat_list = [item for sublist in l for item in sublist]
            flattened_all.append(flat_list)
        self.event_distrib = flattened_all
