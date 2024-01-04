#!/usr/bin/python
import sys


def fun_std(in_array):
    import numpy
    res = numpy.std(in_array)
    return res


def fun_mean(in_array):
    import numpy
    res = numpy.mean(in_array)
    return res


def compute(in_dir, out_file, options):
    import csv
    import glob
    import math
    import numpy
    import matplotlib.pyplot as plt

    file_list = sorted(glob.glob(in_dir + '_g_r_os.csv'))
    print('Files to read: ', file_list)

    all_chunks = []
    chunks_no = []
    timestamps = []
    chunk_averages = []
    active_chunk = None
    first_chunk = None
    timestamp_base = None
    full_chunk = None
    chunk_len = 0

    for file in file_list:
        print('Opening file: %s' % file)
        f = open(file, newline='')
        reader = csv.reader(f, delimiter=options['input_delimiter'])
        if options['headers']:
            next(reader)
        for row in reader:
            chunk = int(row[0])
            timestamp = float(row[1])
            value = float(row[2])

            if active_chunk is None or active_chunk != chunk:
                # Save chunk values
                if full_chunk is not None:
                    active_chunk_average_head = sum(full_chunk[:options['avg_head_len']]) / options['avg_head_len']
                    active_chunk_average_tail = sum(full_chunk[-options['avg_tail_len']:]) / options['avg_tail_len']
                    head_tail_diff = math.fabs(active_chunk_average_head - active_chunk_average_tail)
                    if (options['chunk_drop'] is False or head_tail_diff < options['avg_head_tail_diff']) \
                            and (chunk_len == len(full_chunk) or chunk_len == 0) \
                            and options['filter']['chunk_min'] <= active_chunk <= options['filter']['chunk_max']:
                        print('Saving chunk %s with length %s' % (active_chunk, len(full_chunk)))
                        chunk_averages.append(active_chunk_average_head)
                        all_chunks.append(full_chunk)
                        chunks_no.append(active_chunk)
                        chunk_len = len(full_chunk)
                    else:
                        if chunk_len != len(full_chunk) and chunk_len > 0:
                            print('Chunk %s dropped because of length. Chunk length: %s, should be: %s'
                                  % (active_chunk, len(full_chunk), chunk_len))
                        elif options['filter']['chunk_min'] > active_chunk \
                                or active_chunk > options['filter']['chunk_max']:
                            print('Chunk %s dropped because of filter'
                                  % active_chunk)
                        else:
                            print('Chunk %s dropped because of difference of head and tail averages. Difference: %s'
                                  % (active_chunk, head_tail_diff))

                active_chunk = chunk
                timestamp_base = timestamp
                full_chunk = []

                if first_chunk is None:
                    first_chunk = active_chunk

            # Creating list of timestamps
            if active_chunk == first_chunk:
                delta_timestamp = round(timestamp - timestamp_base, 2)
                timestamps.append(delta_timestamp)

            # Saving chunk value
            full_chunk.append(value)

    # Save last chunk values
    active_chunk_average_head = sum(full_chunk[:options['avg_head_len']]) / options['avg_head_len']
    active_chunk_average_tail = sum(full_chunk[-options['avg_tail_len']:]) / options['avg_tail_len']
    head_tail_diff = math.fabs(active_chunk_average_head - active_chunk_average_tail)
    if not ((options['chunk_drop'] is True and head_tail_diff > options['avg_head_tail_diff'])
            or (chunk_len != len(full_chunk) and chunk_len > 0))\
            and options['filter']['chunk_min'] <= active_chunk <= options['filter']['chunk_max']:
        print('Saving chunk %s with length %s' % (active_chunk, len(full_chunk)))
        chunk_averages.append(active_chunk_average_head)
        all_chunks.append(full_chunk)
        chunks_no.append(active_chunk)
    else:
        if chunk_len != len(full_chunk) and chunk_len > 0:
            print('Chunk %s dropped because of length. Chunk length: %s, should be: %s'
                  % (active_chunk, len(full_chunk), chunk_len))
        elif options['filter']['chunk_min'] > active_chunk or active_chunk > options['filter']['chunk_max']:
            print('Chunk %s dropped because of filter'
                  % active_chunk)
        else:
            print('Chunk %s dropped because of difference of head and tail averages. Difference: %s'
                  % (active_chunk, head_tail_diff))

    # Save values minus average of first 3 values in chunk
    all_chunks2 = [[round(value - chunk_averages[ind], 5) for value in row] for ind, row in enumerate(all_chunks)]

    all_functions = []
    for filter_function in options['functions']:
        all_functions.append([filter_function['callback'](chunk) for chunk in all_chunks2])

    print(all_functions)

    all_chunks_filtered = []
    chunks_no_filtered = []
    for ind, chunk in enumerate(all_chunks2):
        append = True
        for ind2, function in enumerate(all_functions):
            if 'filter' in options['functions'][ind2]:
                append = append and math.fabs(function[ind]) < options['functions'][ind2]['filter']
            if 'filter_min' in options['functions'][ind2]:
                append = append and function[ind] >= options['functions'][ind2]['filter_min']
            if 'filter_max' in options['functions'][ind2]:
                append = append and function[ind] <= options['functions'][ind2]['filter_max']
        if append:
            all_chunks_filtered.append(chunk)
            chunks_no_filtered.append(chunks_no[ind])

    # Compute average for timestamps
    times_averages = [round(float(sum(col)) / len(col), 5) for col in zip(*all_chunks)]
    times_averages2 = [round(float(sum(col)) / len(col), 5) for col in zip(*all_chunks2)]
    times_averages_filetered = [round(float(sum(col)) / len(col), 5) for col in zip(*all_chunks_filtered)]

    print('all_chunks2', len(all_chunks2))
    print('all_chunks_filtered', len(all_chunks_filtered))
    print('times_averages', len(times_averages))
    print('times_averages2', len(times_averages2))
    print('times_averages_filetered', len(times_averages_filetered))
    print('timestamps', len(timestamps))

    print("Writing data to ", out_file + '_processed.csv')
    with open(out_file + '_processed.csv', "w", newline='') as csv_file:
        if options['transpose_results']:
            result1 = [['timestamp\\chunk'] + timestamps + ['avg (' + str(options['avg_head_len']) + ')']] \
                      + [[chunks_no[ind]] + row + [chunk_averages[ind]] for ind, row in enumerate(all_chunks)] \
                      + [['average'] + times_averages + ['']]
            result = numpy.transpose(result1)
        else:
            result = [['chunk\\timestamp'] + timestamps + ['avg (' + str(options['avg_head_len']) + ')']] \
                     + [[chunks_no[ind]] + row + [chunk_averages[ind]] for ind, row in enumerate(all_chunks)] \
                     + [['average'] + times_averages + ['']]

        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(result)

    print("Writing data to ", out_file + '_averaged.csv')
    with open(out_file + '_averaged.csv', "w", newline='') as csv_file:
        if options['transpose_results']:
            result1 = [['timestamp\\chunk'] + timestamps] \
                      + [[chunks_no[ind]] + row for ind, row in enumerate(all_chunks2)] \
                      + [['average'] + times_averages2]
            result = numpy.transpose(result1)
        else:
            result = [['chunk\\timestamp'] + timestamps] \
                     + [[chunks_no[ind]] + row for ind, row in enumerate(all_chunks2)] \
                     + [['average'] + times_averages2]

        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(result)

    print("Writing data to ", out_file + '_filtered.csv')
    with open(out_file + '_filtered.csv', "w", newline='') as csv_file:
        if options['transpose_results']:
            result1 = [['timestamp\\chunk'] + timestamps] \
                      + [[chunks_no_filtered[ind]] + row for ind, row in enumerate(all_chunks_filtered)] \
                      + [['average'] + times_averages_filetered]
            result = numpy.transpose(result1)
        else:
            result = [['chunk\\timestamp'] + timestamps] \
                     + [[chunks_no_filtered[ind]] + row for ind, row in enumerate(all_chunks_filtered)] \
                     + [['average'] + times_averages_filetered]

        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(result)

    print("Ploting to ", out_file + "_plot.png")
    if (options['plot']['draw_first'] or options['plot']['draw_second']) \
            and (options['plot']['draw_points'] or options['plot']['draw_lines']):
        if options['plot']['draw_points']:
            if options['plot']['draw_first']:
                plt.plot(timestamps, times_averages, 'bs')
            if options['plot']['draw_second']:
                plt.plot(timestamps, times_averages2, 'ro')

        if options['plot']['draw_lines']:
            if options['plot']['draw_first']:
                plt.plot(timestamps, times_averages, 'b')
            if options['plot']['draw_second']:
                plt.plot(timestamps, times_averages2, 'r')

        if options['plot']['draw_labels']:
            for i, timestamp in enumerate(timestamps):
                x = timestamp + 0.1
                y = times_averages[i] + 0.1
                y2 = times_averages2[i] - 0.1
                if options['plot']['draw_first']:
                    plt.text(x, y, round(times_averages[i], 3), fontsize=9)
                if options['plot']['draw_second']:
                    plt.text(x, y2, round(times_averages2[i], 3), fontsize=9)

        plt.xlabel('timestamp')
        plt.ylabel('values')
        plt.grid(True)
        plt.savefig(out_file + "_plot.png")

        for row in all_chunks2:
            plt.plot(timestamps, row)

        plt.plot(timestamps, times_averages2, 'r')
        plt.savefig(out_file + "_plot2.png")


def main(argv):
    import sys

    directory = 'Z:\\Data\\2023_06_09_brain\\Z\\sig\\brain6_25BW_700\\'
    result_file = 'result_os'
    options = {
        'headers': True,
        'input_delimiter': ';',
        'avg_head_len': 1000,
        'avg_tail_len': 2000,
        'avg_head_tail_diff': 1,
        'chunk_drop': False,
        'transpose_results': True,
        'plot': {
            'draw_first': False,
            'draw_second': True,
            'draw_labels': False,
            'draw_lines': True,
            'draw_points': False
        },
        'filter': {
            'chunk_min': 0,
            'chunk_max': 600,
        },
        'functions': [
            {
                'callback': fun_std,
                'filter': 0.5
            },
            {
                'callback': fun_mean,
                'filter_min': 0,
                'filter_max': 1
            }
        ]
    }

    try:
        compute(directory, result_file, options)
        # compute(sample_file, result_file, options)
    except (IOError, OSError) as e:
        print(e)
        sys.exit(e.errno)
    except Exception as e:
        print(e)
        sys.exit(-1)


if __name__ == "__main__":
    main(sys.argv[1:])
