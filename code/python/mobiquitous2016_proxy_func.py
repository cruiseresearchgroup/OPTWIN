from windowing.sliding_window import DStreamManager, CsvDsStream
import numpy as np

class MobiQuitous2016ProxyFunc(object):
    def __init__(self, streamdir, filetoinspect):
        self.streamdir = streamdir
        self.filetoinspect = filetoinspect

    def temp_summary_func(self, feature_vector):
        return [
            np.mean(feature_vector),
            np.median(feature_vector),
            np.max(feature_vector),
            np.min(feature_vector),
            np.std(feature_vector),
            np.percentile(feature_vector, 75) - np.percentile(feature_vector, 25),
            np.sqrt(np.mean(np.power(feature_vector, 2)))
        ]

    def temp_summary_func_header(self):
        return [
            'mean',
            'median',
            'max',
            'min',
            'stdv',
            'iqr',
            'rms'
        ]

    def transform_array_nan_to_blank(array_input):
        array_ = np.array(array_input)
        array_.astype(str)

        # Replace nan by white spaces
        array_[array_ == 'nan'] = ''
        return array_

    def streamcreator(self):
        stream_manager = DStreamManager()
        print('created for ' + self.filetoinspect)
        stream_manager.register(CsvDsStream(self.streamdir + self.filetoinspect), rules={
            'default': {
                'func_per_feature': self.temp_summary_func,
                'func_per_feature_headers': self.temp_summary_func_header
            }
        })

        return stream_manager

    def retrieve(self):
        return self.streamcreator