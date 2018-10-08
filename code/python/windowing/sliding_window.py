#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:58:24 2017

@author: jonathanliono
"""

import csv
import numpy as np
from abc import ABCMeta, abstractmethod


# defining object for data stream
class CsvDsStream(object):
    def __init__(self,
                 thefile,
                 hasheader=True,
                 ignored_headers=[],
                 pick_only_headers=[],
                 read_filters={},
                 trailing_f_name=''):
        self.f = open(thefile, newline='')
        reader = csv.reader(self.f)
        self.filename = thefile
        self.headers = next(reader)
        self.filestream = reader
        self.headrow = None
        self.featureheaders = None
        self.ignored_headers = ignored_headers
        self.pick_only_headers = pick_only_headers
        self.read_filters = read_filters
        self.trailing_f_name = trailing_f_name
        self.window_buffer = []  # format of object is {'t':time, 'row': row}
        self.active = True
        self.t_key = None
        self.c_key = None

    def time_key(self):
        return self.headers[0] if self.t_key is None else self.headers[self.t_key]

    def class_label_key(self):
        return self.headers[len(self.headers) - 1] if self.c_key is None else self.headers[self.c_key]

    def set_time_key(self, time_key_to_set):
        isset = False
        for k, i in enumerate(self.headers):
            if i == time_key_to_set:
                self.t_key = k
                isset = True
                break

        return isset

    def set_class_label_key(self, class_label_key_to_set):
        isset = False
        for k, i in enumerate(self.headers):
            if i == class_label_key_to_set:
                self.c_key = k
                isset = True
                break

        return isset

    def next(self):
        nextrow = next(self.filestream)
        kv = {}
        for k, i in enumerate(self.headers):
            kv[i] = nextrow[k]

        return kv

    def close_stream(self):
        self.active = False
        self.f.close()

    def head_row(self):
        if self.headrow is None:
            self.headrow = self.next()

        return self.headrow

    def is_inactive(self):
        return not self.active

    def clear_head(self):
        self.headrow = None

    def feature_headers(self):
        if self.featureheaders is None:
            self.featureheaders = []
            for index, key_header in enumerate(self.headers):
                if (key_header != self.class_label_key()
                        and key_header != self.time_key()
                        and key_header not in self.ignored_headers):
                    if len(self.pick_only_headers) > 0:
                        if key_header in self.pick_only_headers:
                            self.featureheaders.append(key_header)
                    else:
                        self.featureheaders.append(key_header)

        return self.featureheaders

    def add_itemrow_to_window_buffer(self, time, row, label):
        self.window_buffer.append({'t': time, 'row': row, 'label': label})

    def get_rows_from_window_buffer_after(self, time):
        rows = []
        labels = []
        for i in range(0, len(self.window_buffer)):
            row = self.window_buffer[i]
            if row['t'] >= time:
                rows.append(row['row'])
                labels.append(row['label'])

        return rows, labels

    def remove_rows_in_window_buffer_before(self, time):
        rows = []
        for i in range(0, len(self.window_buffer)):
            row = self.window_buffer[i]
            if row['t'] >= time:
                rows.append(row)

        self.window_buffer = rows


class DStreamFeatureConstructionInstanceLvlProxy(metaclass=ABCMeta):
    @abstractmethod
    def construct(self, instance_row): pass


class DStreamManager(object):
    def __init__(self):
        self.streams = []
        self.streams_func_summary_features = []
        self.streams_feature_constructions_instance_lvl = []
        self.starttime = None
        self.windowsize = None
        self.windowstep = None

    def is_inactive(self):
        isallinactive = True
        for stream in self.streams:
            if not stream.is_inactive():
                isallinactive = False
                break

        return isallinactive

    def register(self, ds_stream, rules=None, f_constructions=None):
        """

        :param ds_stream: data stream
        :param rules: key value pair where key "default" is the default one and value should be object that has attribute func_per_feature and func_per_feature_headers
        :param f_constructions: key value pair for constructing features from features of ds_stream param.
                                The key should be the name of new constructed feature and the value should be the function
                                with np.array(). The return value of the function would then be the list of new constructed array
        """
        self.streams.append(ds_stream)
        self.streams_func_summary_features.append(rules)
        self.streams_feature_constructions_instance_lvl.append(f_constructions)

    def define_temporal_sliding_window(self, windowsize=100, step=100):
        self.windowsize = np.float128(windowsize)
        self.windowstep = np.float128(step)
        self.starttime = None

    def temporal_segmentation(self, func_per_feature=None, func_per_feature_headers=None):
        try:
            self.windowsize
            self.windowstep
            self.starttime
        except (AttributeError, NameError):
            print("NOT DEFINED")
            return

        if self.starttime is None:
            for stream in self.streams:
                time = np.float128(stream.head_row()[stream.time_key()])
                if self.starttime is None or self.starttime > time:
                    self.starttime = np.float128(time)
                time = None

        # set expected end time
        endtime = self.starttime + self.windowsize
        #        print('Start time: ')
        #        print(repr(self.starttime))
        #        print('End time: ')
        #        print(endtime)
        row_headers = []
        row_result = []
        row_result_labels = []
        all_class_labels = []  # just for keeping track all occurance of labels.
        for idx_stream, stream in enumerate(self.streams):
            cachedrows, cachedlabels = stream.get_rows_from_window_buffer_after(self.starttime)
            if len(cachedrows) == 0:
                clmncnt = len(stream.feature_headers())
                if self.streams_feature_constructions_instance_lvl[idx_stream] is not None \
                        and len(self.streams_feature_constructions_instance_lvl[idx_stream]) > 0:
                    clmncnt += len(self.streams_feature_constructions_instance_lvl[idx_stream])
                stream_array = np.array(cachedrows).reshape(0, clmncnt)
            else:
                stream_array = np.array(cachedrows)
            stream_classlabel_array = cachedlabels
            itemrow = None

            if stream.is_inactive():
                continue

            while itemrow is None:
                try:
                    itemrow = stream.head_row()
                except StopIteration:
                    stream.active = False
                    break
                time = np.float128(itemrow[stream.time_key()])

                if time >= self.starttime and time < endtime:
                    row = []
                    labels = []

                    # filter mechanism
                    if len(stream.read_filters) > 0:
                        should_continue = False
                        for fkey, fval in stream.read_filters.items():
                            if itemrow[fkey] != fval:
                                should_continue = True
                                break

                        if should_continue:
                            itemrow = None
                            stream.clear_head()
                            continue

                    for key_header in stream.headers:
                        if key_header == stream.class_label_key():
                            labels.append(itemrow[key_header])
                        elif (key_header != stream.time_key()
                              and key_header not in stream.ignored_headers):
                            if len(stream.pick_only_headers) > 0:
                                if key_header in stream.pick_only_headers:
                                    row.append(np.float128(itemrow[key_header] if itemrow[key_header] else np.nan))
                            else:
                                row.append(np.float128(itemrow[key_header] if itemrow[key_header] else np.nan))

                    if self.streams_feature_constructions_instance_lvl[idx_stream] is not None \
                            and len(self.streams_feature_constructions_instance_lvl[idx_stream]) > 0:
                        for fckey, fcfunc in self.streams_feature_constructions_instance_lvl[idx_stream].items():
                            if fcfunc is not None:
                                # row.append(fcfunc.construct(itemrow))  # for implementation of proxy abstract class
                                row.append(fcfunc(itemrow))
                            else:
                                row.append(np.nan)

                    stream_array = np.append(stream_array, np.array([row]), axis=0)
                    unique, pos = np.unique(np.array(labels), return_inverse=True)
                    counts = np.bincount(pos)
                    maxpos = counts.argmax()
                    majoritylabel = unique[maxpos]
                    stream_classlabel_array.append(majoritylabel)
                    all_class_labels.append(majoritylabel)
                    stream.add_itemrow_to_window_buffer(time, row, majoritylabel)
                    itemrow = None
                    stream.clear_head()

            if stream.is_inactive():
                continue

            rules = self.streams_func_summary_features[idx_stream]

            if rules is not None:
                defaultrule = rules['default']
                customrulekeys = list(rules['custom'].keys()) if 'custom' in rules else []
                if defaultrule['func_per_feature'] is not None and defaultrule['func_per_feature_headers'] is not None:
                    feature_headers = list(stream.feature_headers())  # get a copy of feature headers rather than modifying it.
                    if self.streams_feature_constructions_instance_lvl[idx_stream] is not None \
                            and len(self.streams_feature_constructions_instance_lvl[idx_stream]) > 0:
                        feature_headers += list(self.streams_feature_constructions_instance_lvl[idx_stream].keys())

                    for index, key_header in enumerate(feature_headers):
                        incustomrule = any(key_header in s for s in customrulekeys)
                        feature_matrix = stream_array[:, index]

                        summary = defaultrule['func_per_feature'](feature_matrix) if not incustomrule else \
                            rules['custom'][key_header]['func_per_feature'](feature_matrix)
                        row_result.extend(summary)
                        headernamestouse = defaultrule['func_per_feature_headers']() if not incustomrule else \
                            rules['custom'][key_header]['func_per_feature_headers']()
                        for header_summary in headernamestouse:
                            trailingname = '' if stream.trailing_f_name == '' else '_' + str(stream.trailing_f_name)
                            row_headers += [key_header + trailingname + '_' + header_summary]

            else:
                if func_per_feature is not None and func_per_feature_headers is not None:
                    feature_headers = stream.feature_headers()
                    if self.streams_feature_constructions_instance_lvl[idx_stream] is not None \
                            and len(self.streams_feature_constructions_instance_lvl[idx_stream]) > 0:
                        feature_headers += list(self.streams_feature_constructions_instance_lvl[idx_stream].keys())

                    for index, key_header in enumerate(feature_headers):
                        feature_matrix = stream_array[:, index]
                        summary = func_per_feature(feature_matrix)
                        row_result.extend(summary)
                        for header_summary in func_per_feature_headers():
                            trailingname = '' if stream.trailing_f_name == '' else '_' + str(stream.trailing_f_name)
                            row_headers += [key_header + trailingname + '_' + header_summary]

            stream_classlabel_array = list(filter(''.__ne__, stream_classlabel_array))
            if len(stream_classlabel_array) > 0:
                unique, pos = np.unique(np.array(stream_classlabel_array), return_inverse=True)
                counts = np.bincount(pos)
                maxpos = counts.argmax()
                majoritylabel = unique[maxpos]
                row_result_labels.append(majoritylabel)

        self.starttime = self.starttime + self.windowstep
        for stream in self.streams:
            stream.remove_rows_in_window_buffer_before(self.starttime)

        if len(row_result_labels) > 0:
            unique, pos = np.unique(np.array(row_result_labels), return_inverse=True)
            counts = np.bincount(pos)
            maxpos = counts.argmax()
            majoritylabel = unique[maxpos]
            if isinstance(majoritylabel, list):
                majoritylabel = majoritylabel[0]
        else:
            majoritylabel = ''

        # check row result before returning
        if (len(row_result) == 0 or all(np.isnan(value) for value in row_result)) and majoritylabel == '':
            if self.is_inactive():
                raise TimeoutError('no more to segment')
            else:
                return None, row_headers, None, []

        return row_result, row_headers, majoritylabel, all_class_labels
