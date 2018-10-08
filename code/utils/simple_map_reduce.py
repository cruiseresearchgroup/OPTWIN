import collections
import itertools
import multiprocessing


class SimpleMapReduce(object):
    def __init__(self, map_func, reduce_func, num_workers=None):
        """
        map_func

          Function to map inputs to intermediate data. Takes as
          argument one input value and returns a tuple with the key
          and a value to be reduced.

        reduce_func

          Function to reduce partitioned version of intermediate data
          to final output. Takes as argument a key as produced by
          map_func and a sequence of the values associated with that
          key.

        num_workers

          The number of workers to create in the pool. Defaults to the
          number of CPUs available on the current host.
        """
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.num_workers = num_workers

    def partition(self, mapped_values):
        """Organize the mapped values by their key.
        Returns an unsorted sequence of tuples with a key and a sequence of values.
        """
        partitioned_data = collections.defaultdict(list)
        for key, value in mapped_values:
            partitioned_data[key].append(value)
        return partitioned_data.items()

    def __call__(self, inputs, chunksize=1):
        """Process the inputs through the map and reduce functions given.

        inputs
          An iterable containing the input data to be processed.

        chunksize=1
          The portion of the input data to hand to each worker.  This
          can be used to tune performance during the mapping phase.
        """
        try:
            pool = multiprocessing.Pool(self.num_workers)
            map_responses = pool.map(self.map_func, inputs, chunksize=chunksize)
            partitioned_data = self.partition(itertools.chain(*map_responses))
            reduced_values = pool.map(self.reduce_func, partitioned_data)
        finally:
            pool.close()
            pool.join()
            pool.terminate()
            pool = None
        return reduced_values
