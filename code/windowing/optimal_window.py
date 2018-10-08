import os
from os import listdir
import pathlib
from os.path import isfile, isdir, join

import csv
import numpy as np
import json

import time

# defining object for data stream
from utils.max_divergence_processor import AsymmetricMaxDivergenceProcessor
from utils.simple_map_reduce import SimpleMapReduce
from windowing.sliding_window import DStreamManager, CsvDsStream

from pickle import dumps, loads


class MapReduceForFeatureExtraction(object):
    def __init__(self, cache_folder, clear_cache, func_datasource_arr, overlap_percentage, label_separator):
        self.cache_folder = cache_folder
        self.clear_cache = clear_cache
        self.func_datasource_arr = func_datasource_arr
        self.overlap_percentage = overlap_percentage
        self.label_separator = label_separator

    def check_all_item_equal(self, lst):
        return lst[1:] == lst[:-1]

    def map_operation(self, window_size):
        output = []
        if self.clear_cache \
                or (not self.clear_cache
                    and (
                                not os.path.exists(join(self.cache_folder, 'f_cached_' + str(window_size) + '.csv'))
                            or not os.path.exists(join(self.cache_folder, 'pi_cached_' + str(window_size) + '.csv'))
                    )):
            dimension_of_label = -1
            dimension_of_optwin = -1
            sliding_window_step = window_size - (self.overlap_percentage * window_size)
            print('valid for {} window size with sliding window step of {}'.format(window_size, sliding_window_step))
            mimpure_arr = []
            msegment = 0
            # feature_instances = None
            labelset = []
            labeltsetindex = []
            cachedfilename = join(self.cache_folder, 'f_cached_' + str(window_size) + '.csv')
            cachedfile = open(cachedfilename, 'w', encoding='utf8')
            filewriterForCachedFile = csv.writer(cachedfile, delimiter=',',
                                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
            haswrittenheader = False
            try:
                for func_datasource in self.func_datasource_arr:
                    stream_manager = func_datasource.retrieve()()
                    stream_manager.define_temporal_sliding_window(window_size, sliding_window_step)

                    while not stream_manager.is_inactive():
                        try:
                            (summary, row_headers, label, all_class_labels) = stream_manager.temporal_segmentation()
                            row_headers += ['label']
                            if not haswrittenheader:
                                filewriterForCachedFile.writerow(row_headers)
                                haswrittenheader = True

                            if summary is not None and len(all_class_labels) > 0:
                                summary.append(label)
                                filewriterForCachedFile.writerow(summary)

                                dimension_of_label, dimension_of_optwin, mimpure_arr = self.initialise_optwin_metrics_for_current_window_size(
                                    label, dimension_of_label, dimension_of_optwin, mimpure_arr)

                                split_labels = label.split(self.label_separator)
                                if len(labelset) == 0 or len(labeltsetindex) == 0:
                                    for il in range(0, len(split_labels)):
                                        labelset.append([])
                                        labeltsetindex.append({})

                                    if len(split_labels) > 1:
                                        labelset.append([])
                                        labeltsetindex.append({})

                                # if len(split_labels) > 1:
                                #     split_labels.append(label)

                                for il in range(0, len(split_labels)):
                                    if split_labels[il] not in labeltsetindex[il]:
                                        value_of_class_index = len(labeltsetindex[il]) + 1
                                        labeltsetindex[il][str(split_labels[il])] = value_of_class_index

                                    labelset[il].append(labeltsetindex[il][str(split_labels[il])])

                                if len(split_labels) > 1:
                                    il = dimension_of_optwin - 1
                                    if label not in labeltsetindex[il]:
                                        value_of_class_index = len(labeltsetindex[il]) + 1
                                        labeltsetindex[il][str(label)] = value_of_class_index

                                    labelset[il].append(labeltsetindex[il][str(label)])

                                msegment += 1

                                for z in range(0, dimension_of_label):
                                    if not self.check_all_item_equal(
                                            [i.split(self.label_separator)[z] for i in all_class_labels]):
                                        mimpure_arr[z] += 1

                                if dimension_of_label > 1:
                                    if not self.check_all_item_equal(all_class_labels):
                                        mimpure_arr[dimension_of_optwin - 1] += 1

                        except TimeoutError as error:
                            print('End operation due to: ' + repr(error))

                    for file_stream in stream_manager.streams:
                        # print(file_stream.next())
                        file_stream.close_stream()
            finally:
                filewriterForCachedFile = None
                cachedfile.close()
                cachedfile = None

            # transpose before writing to csv
            labelsettosave = np.transpose(np.array(labelset, dtype=np.int16))
            np.savetxt(join(self.cache_folder, 'ls_cached_' + str(window_size) + '.csv'), labelsettosave, delimiter=",",
                       fmt='%i')

            with open(join(self.cache_folder, 'ls_cached_' + str(window_size) + '.json'), 'w') as outfile:
                json.dump(labeltsetindex, outfile)

            # compute proportion impurity
            for metric_dimension in range(0, len(mimpure_arr)):
                mimpure_arr[metric_dimension] = mimpure_arr[metric_dimension] / msegment

            np.savetxt(join(self.cache_folder, 'pi_cached_' + str(window_size) + '.csv'), mimpure_arr, delimiter=",")
            # print('window size: {}'.format(i))
            # print(optwin_map[str

            output.append(('total_processes', 1))
        else:
            output.append(('total_processes', 0))

        return output

    def initialise_optwin_metrics_for_current_window_size(self, label, dimension_of_label,
                                                          dimension_of_optwin, mimpure_arr):
        split_labels = label.split(self.label_separator)

        if dimension_of_label == -1 and dimension_of_optwin == -1:
            dimension_of_label = len(split_labels)
            dimension_of_optwin = dimension_of_label
            if dimension_of_label > 1:
                dimension_of_optwin += 1

        if dimension_of_optwin != -1 and len(mimpure_arr) == 0:
            mimpure_arr = [0] * dimension_of_optwin

        return dimension_of_label, dimension_of_optwin, mimpure_arr

    def reduce_operation(self, item):
        key, values = item
        return (key, sum(values))

    def start_operation(self, window_sizes):
        mapper = SimpleMapReduce(self.map_operation, self.reduce_operation)
        value = mapper(window_sizes)
        return value[1] if value[0] == 'total_processes' else 0


class CsvOptimalWindowing(object):
    def __init__(self, label_map={}, separator=':'):
        self.label_separator = separator
        print('initialising')

    def extract_features_and_necessary_stats(self, cache_folder, clear_cache, func_datasource_arr, overlap_percentage,
                                             window_end_size, window_start_size, window_step):

        window_sizes_to_parallelise = []

        index_window = window_start_size
        while index_window <= window_end_size:
            window_sizes_to_parallelise.append(index_window)
            index_window += window_step

        mapreduce = MapReduceForFeatureExtraction(cache_folder,
                                                  clear_cache,
                                                  func_datasource_arr,
                                                  overlap_percentage,
                                                  self.label_separator)

        return mapreduce.start_operation(window_sizes_to_parallelise)

    def extract_features(self, window_start_size, window_end_size, window_step,
                         func_datasource_arr,  # a function that should return a new DStreamManager
                         cache_folder,
                         overlap_percentage=0.0,
                         asymmetric_divergence=True, clear_cache=True, milliseconds_precision=True):
        pathlib.Path(cache_folder).mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        print('Begin Feature Extraction process with map reduce... ')
        files_processed = self.extract_features_and_necessary_stats(cache_folder,
                                                                    clear_cache,
                                                                    func_datasource_arr,
                                                                    overlap_percentage,
                                                                    window_end_size,
                                                                    window_start_size,
                                                                    window_step)
        print('FINISHED Feature Extraction process with map reduce... ')
        print("--- %s seconds ---" % (time.time() - start_time))
        return files_processed

    def infer_best_window_size(self, window_start_size, window_end_size, window_step,
                               func_datasource_arr,  # a function that should return a new DStreamManager
                               cache_folder,
                               overlap_percentage=0.0,
                               asymmetric_divergence=True, clear_cache=True):
        optwin_key_indexes = []

        index_window = window_start_size
        while index_window <= window_end_size:
            optwin_key_indexes.append(index_window)
            index_window += window_step

        print(optwin_key_indexes)
        if len(optwin_key_indexes) <= 2:
            raise ValueError('Parameter error... ')

        start_time = time.time()
        # process for necessary metrics computation
        optwin_map = {}

        metric_caches_filename = 'ls_cached_metrics.csv'

        try:
            if clear_cache \
                    or (not clear_cache
                        and (
                                not os.path.exists(join(cache_folder, metric_caches_filename))
                        )):
                print('Begin computing OPTWIN metrics process... ')
                with open(join(cache_folder, metric_caches_filename), 'w', encoding='utf8') as divergence_cache_file:
                    filewriter_divergencecache = csv.writer(divergence_cache_file, delimiter=',',
                                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

                    # format of columns would be: in this format:
                    # 'time_window_size',
                    # 'column_impurity_proportion',
                    # 'column_mean_max_divergence',
                    # ...,
                    # 'column_impurity_proportion_multilabel',
                    # 'column_mean_max_divergence_multilabel'
                    index_window = window_start_size
                    while index_window <= window_end_size:
                        print('Computing OPTWIN metrics process for window size {} seconds... '.format(index_window))
                        row_to_print = [index_window]
                        cachedfilename = join(cache_folder, 'f_cached_' + str(index_window) + '.csv')
                        with open(cachedfilename) as f:
                            ncols = len(f.readline().split(','))

                        feature_instances = np.loadtxt(cachedfilename, delimiter=',', skiprows=1,
                                                       usecols=range(0, ncols - 1))
                        # num_instances, size_of_f = feature_instances.shape
                        # print('size of features {}'.format(size_of_f))

                        labelset = np.genfromtxt(join(cache_folder, 'ls_cached_' + str(index_window) + '.csv'), delimiter=',',
                                                 dtype=np.int16)
                        labelset = np.transpose(labelset)

                        pimpure_arr = np.genfromtxt(join(cache_folder, 'pi_cached_' + str(index_window) + '.csv'), delimiter=',')
                        dimension_of_optwin = len(pimpure_arr)
                        # print('dimension of pimpure: {}'.format(len(pimpure_arr)))
                        if dimension_of_optwin > 1:
                            dimension_of_label = dimension_of_optwin - 1
                            # print('dimension of label: {}'.format(dimension_of_label))

                        with open(join(cache_folder, 'ls_cached_' + str(index_window) + '.json')) as data_file:
                            labeltsetindex = json.load(data_file)

                        labelset = labelset[0:-1, :]
                        del labeltsetindex[-1]
                        max_div_all_labelsets = AsymmetricMaxDivergenceProcessor().calculate_maximum_divergences(
                            features2d=feature_instances,
                            class_values=labelset,
                            unique_class_map=labeltsetindex)
                        mean_maximum_divergence_all_label_sets = []
                        for max_div_label_set in max_div_all_labelsets:
                            mean_maximum_divergence_all_label_sets.append(np.mean(max_div_label_set))

                        optwin_map[str(index_window)] = []
                        for metric_dimension in range(0, dimension_of_label):
                            curr_pimpure = pimpure_arr[metric_dimension]
                            curr_mean_max_divergence = mean_maximum_divergence_all_label_sets[metric_dimension]
                            row_to_print.append(curr_pimpure)
                            row_to_print.append(curr_mean_max_divergence)
                            optwin_map[str(index_window)].append({
                                'impure_proportion': curr_pimpure,
                                'mean_max_divergence': curr_mean_max_divergence
                            })

                        if dimension_of_label > 1:
                            metric_dimension = dimension_of_optwin - 1
                            curr_pimpure = pimpure_arr[metric_dimension]
                            curr_mean_max_divergence = np.mean(mean_maximum_divergence_all_label_sets)
                            row_to_print.append(curr_pimpure)
                            row_to_print.append(curr_mean_max_divergence)
                            optwin_map[str(index_window)].append({
                                'impure_proportion': curr_pimpure,
                                'mean_max_divergence': curr_mean_max_divergence
                            })

                        filewriter_divergencecache.writerow(row_to_print)
                        index_window += window_step
        except OSError as error:
            print('End operation due to: ' + repr(error))
            return -1

        print('Finished computing OPTWIN metrics (impurity proportion and mean max divergence) process... ')
        print("--- %s seconds ---" % (time.time() - start_time))

        print('Begin finding optimal window size process... ')
        # clean up
        feature_instances = None
        labelset = []
        labeltsetindex = []

        # labelset_idx_for_recommendation = 0
        # if dimension_of_label > 1:
        #     labelset_idx_for_recommendation = dimension_of_optwin - 1
        #
        # impure_metrics = []
        # divergences = []
        divergence_deviations = []

        with open(join(cache_folder, metric_caches_filename)) as f:
            ncols = len(f.readline().split(','))

        metrics = np.loadtxt(join(cache_folder, metric_caches_filename), delimiter=',', skiprows=0, usecols=(0, ncols - 2, ncols - 1))

        impure_metrics = metrics[:, 1]
        divergences = metrics[:, 2]
        # for i in range(window_start_size, window_end_size + 1, window_step):
        #     impure_metrics.append(optwin_map[str(i)][labelset_idx_for_recommendation]['impure_proportion'])
        #     divergences.append(optwin_map[str(i)][labelset_idx_for_recommendation]['mean_max_divergence'])

        mean_divergence = np.mean(divergences)

        for i in range(0, len(divergences)):
            divergence_deviations.append(np.power(divergences[i] - mean_divergence, 2))

        min_divergence_metrics = np.min(divergence_deviations)
        max_divergence_metrics = np.max(divergence_deviations)
        if max_divergence_metrics == min_divergence_metrics:
            print('Invalid to find optimal window size')
            return None

        divergence_metrics = []
        for i in range(0, len(divergence_deviations)):
            divergence_metrics.append(
                (divergence_deviations[i] - min_divergence_metrics) / (max_divergence_metrics - min_divergence_metrics))

        for i in range(0, len(impure_metrics) - 1):
            current_pimpure = impure_metrics[i]
            current_divergence = divergence_metrics[i]
            next_pimpure = impure_metrics[i + 1]
            next_divergence = divergence_metrics[i + 1]

            centroid = np.mean([current_pimpure, current_divergence, next_pimpure, next_divergence])

            min_impurity = np.min([current_pimpure, next_pimpure])
            max_impurity = np.max([current_pimpure, next_pimpure])
            min_ndd = np.min([current_divergence, next_divergence])
            max_ndd = np.max([current_divergence, next_divergence])
            # intersect = min_impurity <= centroid <= max_impurity and min_ndd <= centroid <= max_ndd
            intersect = self.overlap(min_impurity, max_impurity, min_ndd, max_ndd)

            if intersect:
                distance_current_impure = abs(centroid - current_pimpure)
                distance_next_impure = abs(next_pimpure - centroid)
                if distance_current_impure < distance_next_impure:
                    return metrics[i, 0]
                else:
                    return metrics[i + 1, 0]
        return -1

    def overlap(self, start1, end1, start2, end2):
        """Does the range (start1, end1) overlap with (start2, end2)?"""
        return not (end1 < start2 or end2 < start1)

    def initialise_and_append_features_instance_to_memory(self, feature_instances, summary):
        if feature_instances is None:
            feature_instances = np.array([]).reshape(0, len(summary))
        return np.append(feature_instances, np.array([summary]), axis=0)
