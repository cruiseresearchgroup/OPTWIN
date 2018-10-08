import csv
import numpy as np
import math
import time

from utils.simple_map_reduce import SimpleMapReduce


class NormalizeMidrange(object):
    __current_range = None
    __current_middle = None

    def __init__(self, middle, the_range):
        self.normal_middle = middle
        self.normal_range = the_range

    def build(self, features2d):
        num_instances, num_of_features = features2d.shape
        feature_stats_max = []  # double, size is n-features
        feature_stats_min = []  # double, size is n-features

        self.__current_range = []  # double, size is n-features
        self.__current_middle = []  # double, size is n-features

        for i in range(0, num_of_features):
            fv = features2d[:, i]  # access by column
            feature_stats_max.append(np.nanmax(fv))
            feature_stats_min.append(np.nanmin(fv))
            self.__current_range.append(feature_stats_max[i] - feature_stats_min[i])
            self.__current_middle.append((feature_stats_min[i] + feature_stats_max[i]) / 2)

    def filter_instance(self, feature_values):
        num_features = len(feature_values)
        histogram_values = []

        for i in range(0, num_features):
            if feature_values[i] is None:
                feature_values[i] = self.normal_middle
                continue

            mid_value = self.__current_middle[i]
            the_range = self.__current_range[i]
            histogram_values.append(
                ((feature_values[i] - mid_value) / the_range) * self.normal_range + self.normal_middle)
            if histogram_values[i] is not None:
                if np.isnan(histogram_values[i]):
                    histogram_values[i] = self.normal_middle
            else:
                histogram_values[i] = self.normal_middle

        return histogram_values

    def filter(self, features2d):
        num_instances, num_features = features2d.shape

        if self.__current_range is None or self.__current_middle is None:
            self.build(features2d)

        for i in range(0, num_instances):
            histogramFeatureValues = self.filter_instance(features2d[i, :])
            features2d[i, :] = np.array(histogramFeatureValues)


class AsymmetricMaxDivergencePairwise(object):
    def __init__(self, p, q, features2d, class_values, bins):
        self.p = p
        self.q = q
        self.features2d = features2d
        self.class_values = class_values
        self.bins = bins

    def pair_wise(self):
        num_instances, num_features = self.features2d.shape
        divergences = [0] * num_features

        # For probability distributions P and Q of a discrete random variable
        # the Kullback-Leibler divergence of Q from P is defined to be:
        # D_KL(P|Q)=sum_i(P(i)log(P(i)/Q(i)))
        max_sum = 0

        for i in range(0, num_features):
            the_sum = 0
            countQ = [0] * self.bins
            countP = [0] * self.bins
            pCount = 0
            qCount = 0
            for j in range(0, num_instances):
                feature_value = self.features2d[j, i]
                if feature_value is None:
                    continue

                if self.class_values[j] == self.q:
                    countQ[int(feature_value)] += 1
                    qCount += 1

                if self.class_values[j] == self.p:
                    countP[int(feature_value)] += 1
                    pCount += 1

            for j in range(0, len(countP)):
                countP[j] /= pCount
                countQ[j] /= qCount

                # Probabilities should never be really 0, they can be small
                if countP[j] == 0:
                    countP[j] = 0.0000001
                if countQ[j] == 0:
                    countQ[j] = 0.0000001

                the_sum += countP[j] * math.log(countP[j] / countQ[j])

            divergences[i] = the_sum
            # Keep track of highest value
            if the_sum > max_sum:
                max_sum = the_sum

        # Normalize to [0,1]
        for i in range(0, num_features):
            divergences[i] /= max_sum

        return divergences


class MapReduceAsymmetricMaxDivergencePairwise(object):
    def map_operation(self, pairwise_object):
        output = []
        divergences = pairwise_object.pair_wise()
        output.append((str(pairwise_object.p) + '-' + str(pairwise_object.q), divergences))
        return output

    def reduce_operation(self, item):
        key, values = item
        return (key, values[0])

    def start_operation(self, pairwise_objects):
        print('[MAX_DIVERGENCE] STARTED MAP REDUCE pairwise divergence computation... ')
        start_time = time.time()
        mapper = SimpleMapReduce(self.map_operation, self.reduce_operation)
        value = mapper(pairwise_objects)
        mapper = None
        print('[MAX_DIVERGENCE] FINISHED MAP REDUCE pairwise divergence computation... ')
        print("--- %s seconds ---" % (time.time() - start_time))
        return value


class AsymmetricMaxDivergenceProcessor(object):
    DEFAULT_BIN = 100

    def __init__(self, label_map={}, separator=':'):
        self.bin = AsymmetricMaxDivergenceProcessor.DEFAULT_BIN

    # returns list of divergence for all label sets
    def calculate_maximum_divergences(self, features2d, class_values, unique_class_map):
        # begin validation
        if features2d is None or class_values is None \
                or len(features2d) == 0 \
                or len(features2d[:, 0]) == 0 \
                or len(class_values) == 0 \
                or len(class_values[0]) == 0 \
                or len(unique_class_map) == 0 \
                or features2d[0] is None \
                or class_values[0] is None \
                or unique_class_map is None \
                or len(class_values) != len(unique_class_map):
            return None

        num_instances, size_of_f = features2d.shape
        num_of_labelsets = len(class_values)
        for j in range(0, num_of_labelsets):
            if num_instances != len(class_values[j]):
                return None

        # end validation

        print('[MAX_DIVERGENCE] Started filtering... ')
        start_time = time.time()
        bins = self.DEFAULT_BIN
        nm = NormalizeMidrange(bins / 2, bins - 0.000001)
        nm.build(features2d)
        nm.filter(features2d)
        print('[MAX_DIVERGENCE] Finished filtering... ')
        print("--- %s seconds ---" % (time.time() - start_time))

        max_divergences_all_label_sets = []

        print('DIMENSION of class values: {}'.format(len(class_values)))

        for i in range(0, len(class_values)):
            # calculate max divergences of all features, returns array
            max_divergences_of_features = self.calculate_max_divergence(features2d, class_values[i],
                                                                        unique_class_map[i],
                                                                        size_of_f, bins)
            max_divergences_all_label_sets.append(max_divergences_of_features)

        return max_divergences_all_label_sets

    def calculate_max_divergence(self, features2d, class_values, unq_dominant_label_index, size_of_f, bins):
        max_divergences = [0] * size_of_f
        pairwisedivergence = {}
        unq_lbl_values = list(unq_dominant_label_index.values())

        items_to_parallelised = []
        for p in unq_lbl_values:
            for q in unq_lbl_values:
                if p != q:
                    items_to_parallelised.append(AsymmetricMaxDivergencePairwise(p, q, features2d, class_values, bins))
                    # d = self.pair_wise(p, q, features2d, class_values, bins)
                    # # put pairwise divergence into 2d map
                    # pairwisedivergence[str(p) + '-' + str(q)] = d

        results = MapReduceAsymmetricMaxDivergencePairwise().start_operation(
            pairwise_objects=items_to_parallelised)
        for key, value in results:
            pairwisedivergence[key] = value

        for p in unq_lbl_values:
            for q in unq_lbl_values:
                key = str(p) + '-' + str(q)
                if key in pairwisedivergence:
                    d = pairwisedivergence[key]
                    if d is not None:
                        for i in range(0, len(d)):
                            if d[i] > max_divergences[i]:
                                max_divergences[i] = d[i]

        return max_divergences

        # def pair_wise(self, p, q, features2d, class_values, bins):
        #     num_instances, num_features = features2d.shape
        #     divergences = [0] * num_features
        #
        #     # For probability distributions P and Q of a discrete random variable
        #     # the Kullback-Leibler divergence of Q from P is defined to be:
        #     # D_KL(P|Q)=sum_i(P(i)log(P(i)/Q(i)))
        #     max_sum = 0
        #
        #     for i in range(0, num_features):
        #         the_sum = 0
        #         countQ = [0] * bins
        #         countP = [0] * bins
        #         pCount = 0
        #         qCount = 0
        #         for j in range(0, num_instances):
        #             feature_value = features2d[j, i]
        #             if feature_value is None:
        #                 continue
        #
        #             if class_values[j] == q:
        #                 countQ[int(feature_value)] += 1
        #                 qCount += 1
        #
        #             if class_values[j] == p:
        #                 countP[int(feature_value)] += 1
        #                 pCount += 1
        #
        #         for j in range(0, len(countP)):
        #             countP[j] /= pCount
        #             countQ[j] /= qCount
        #
        #             # Probabilities should never be really 0, they can be small
        #             if countP[j] == 0:
        #                 countP[j] = 0.0000001
        #             if countQ[j] == 0:
        #                 countQ[j] = 0.0000001
        #
        #             the_sum += countP[j] * math.log(countP[j] / countQ[j])
        #
        #         divergences[i] = the_sum
        #         # Keep track of highest value
        #         if the_sum > max_sum:
        #             max_sum = the_sum
        #
        #     # Normalize to [0,1]
        #     for i in range(0, num_features):
        #         divergences[i] /= max_sum
        #
        #     return divergences
