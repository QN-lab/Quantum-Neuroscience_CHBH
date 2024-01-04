#!/usr/bin/python
fn='ania__33_38_Hz_c1__with_error' #file to read without .csv extension 

class Options:
    def __init__(self):
        self.headers = True
        self.input_delimiter = ','
        self.input_directory = '.\\'
        self.output_name = str(fn)
        self.averages = {
            'head_len': 130, #baseline correction - number of points used to calculate baseline
            'tail_len': 200,
            'drop': False,
            'tail_head_diff': 0.02
        }
        self.averages_for_x_ms = { #this is a new feature i added in order to calculate average inside of each chunk for the first X rows
            'aver_len': 650
            }
        self.filter = {
            'chunk_min': 0,
            'chunk_max': 149,
            'drop': False
            }
        self.functions = [
            {
                'label': 'Standard deviation',
                'callback': fun_std,
                'filter': 800,
                'drop': True,
                'drop_corrected': False
            },
            {
                'label': 'Arithmetic mean',
                'callback': fun_mean,
                'filter_min': -550,
                'filter_max': 550,
                'drop': True, #applies filter to values which are not "corrected" shifted by the initial offset determined by head_len 
                'drop_corrected': False, #filter is applied to the corrected values 
            }
        ]


def fun_std(in_array):
    import numpy
    res = numpy.std(in_array)
    return res


def fun_mean(in_array):
    import numpy
    res = numpy.mean(in_array)
    return res


class Chunk:
    def __init__(self):
        self.number = None
        self.base_timestamp = None
        self.timestamps = []
        self.values = []
        self.corrected_values = []

        self.dropped = False
        self.head_average = None
        self.tail_average = None
        self.average_for_x_ms = None
        self.difference = None

        self.functions = []
        self.functions_corrected = []
        self.mean_for_x_ms = []

    def add_row(self, number: int, value: float, timestamp: float) -> bool:
        if self.number is not None and self.number != number:
            self.compute_averages()
            return False
        self.number = number
        if self.base_timestamp is None:
            self.base_timestamp = timestamp
        self.timestamps.append(timestamp - self.base_timestamp)
        self.values.append(value)
        #print(self.base_timestamp,'.......')
       # print(self.timestamps,'***')
        return True
        
    def compute_averages(self):
        import math
        self.head_average = sum(self.values[:options.averages['head_len']]) / options.averages['head_len']
        self.tail_average = sum(self.values[-options.averages['tail_len']:]) / options.averages['tail_len']
        self.difference = self.head_average - self.tail_average
       # print(self.head_average, self.tail_average, self.difference)
        #print(self.tail_average)

        if options.averages['drop'] \
                and math.fabs(self.head_average - self.tail_average) > options.averages['tail_head_diff']:
            #print('Chunk %s was dropped because of diff in average of head and tail' % self.number)
            self.dropped = True
   
    def compute_averages_for_x_ms(self):
        import math
        
        self.average_for_x_ms = sum(self.values[:options.averages_for_x_ms['aver_len']]) / options.averages_for_x_ms['aver_len']
        
       # print('average of chunk %s over '  % self.number, options.averages_for_x_ms['aver_len'], 'ms', self.average_for_x_ms)

        #print(self.mean_for_x_ms)
        
    def compute(self):
        import math
        # Baseline correction: Computing corrected values - reduce values by head average
        
        if self.head_average is None:
            self.compute_averages()
            
        if self.average_for_x_ms is None:     
            self.compute_averages_for_x_ms()
        
        self.corrected_values = [round(val - self.head_average, 5) for val in self.values]

        for fun in options.functions:
            fun_val = fun['callback'](self.values)
            if fun['drop']:
                if 'filter' in fun and math.fabs(fun_val) > fun['filter'] and not self.dropped:
                   # print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
                if 'filter_min' in fun and fun_val < fun['filter_min'] and not self.dropped:
                   # print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
                if 'filter_max' in fun and fun_val > fun['filter_max'] and not self.dropped:
                   # print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
            self.functions.append(fun_val)

            fun_val = fun['callback'](self.corrected_values)
            if fun['drop_corrected']:
                if 'filter' in fun and math.fabs(fun_val) > fun['filter'] and not self.dropped:
                 #   print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
                if 'filter_min' in fun and fun_val < fun['filter_min'] and not self.dropped:
                  #  print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
                if 'filter_max' in fun and fun_val > fun['filter_max'] and not self.dropped:
                   # print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
            self.functions_corrected.append((fun['callback'](self.corrected_values)))

    def to_array(self):
        return self.values

    def to_array_corrected(self):
        return self.corrected_values


class Data:
    def __init__(self):
        self.chunks = []
        self.chunks_length = None
        self.timestamps_averages = []
        self.timestamps_averages_corrected = []

    def add_chunk(self, chunk):
        chunk_len = len(chunk.values)
        if self.chunks_length is not None and self.chunks_length != chunk_len and not chunk.dropped:
            print('Chunk no %s marked dropped because of difference in chunk length. Is %s, should be %s.'
                  % (chunk.number, chunk_len, self.chunks_length))
            chunk.dropped = True

        if False and self.chunks_length is not None and self.chunks[0].timestamps != chunk.timestamps and not chunk.dropped:
            print('Chunk no %s marked dropped because of difference in timestamps.' % chunk.number)
            chunk.dropped = True

        if not chunk.dropped \
                and (chunk.number < options.filter['chunk_min'] or chunk.number > options.filter['chunk_max']) \
                and options.filter['drop']:
            #print('Chunk no %s marked dropped because of difference filter.' % chunk.number)
            chunk.dropped = True

        if self.chunks_length is None:
            self.chunks_length = chunk_len

        #print('Saving chunk no: %s' % chunk.number)
        self.chunks.append(chunk)
        
    def compute_chunks(self):
        import numpy 
        
        for chunk in self.chunks:
            chunk.compute()

        temp_chunks = [chunk.to_array() for chunk in self.chunks if not chunk.dropped]
        self.timestamps_averages = [round(float(sum(col)) / len(col), 5) for col in zip(*temp_chunks)]
        temp_chunks = [chunk.to_array_corrected() for chunk in self.chunks if not chunk.dropped]
        self.timestamps_averages_corrected = [round(float(sum(col)) / len(col), 5) for col in zip(*temp_chunks)]
       
       
    def save_to_file(self, filename):
        import csv
        import numpy
        import scipy

        with open(filename + '_all.csv', "w", newline='') as csv_file:
            result_1 = [['timestamps\\chunk'] + self.chunks[0].timestamps + [fun['label'] for fun in options.functions]] \
                        + [[row.number] + row.values + row.functions for row in self.chunks if not row.dropped] \
                        + [['average'] + self.timestamps_averages + ['' for fun in options.functions]]
            result = numpy.transpose(result_1)
            print('chunks saved:', numpy.size(result,1)-2)
            print('whole matrix',numpy.size(result))
            noise_pkpk=numpy.amax(self.timestamps_averages,axis=0)-numpy.amin(self.timestamps_averages,axis=0)
            print('noise_pk_pk:',noise_pkpk)
            print('noise rms (1 sdev):', numpy.std(self.timestamps_averages, axis=0))
            print('noise mean:', numpy.mean(self.timestamps_averages, axis=0))
            from scipy import stats
            print('noise stdev of the mean (SEM):', scipy.stats.sem(self.timestamps_averages, axis=0, ddof=0))
            writer = csv.writer(csv_file, delimiter = ',')
            writer.writerows(result)

        with open(filename + '_corrected.csv', "w", newline='') as csv_file:
            result_1 = [['timestamps\\chunk'] + self.chunks[0].timestamps + [fun['label'] for fun in options.functions]] \
                        + [[row.number] + row.corrected_values + row.functions_corrected for row in self.chunks if not row.dropped] \
                        + [['average'] + self.timestamps_averages_corrected + ['' for fun in options.functions]]
            result = numpy.transpose(result_1)

            writer = csv.writer(csv_file, delimiter = ',')
            writer.writerows(result)
        ##############################
            #tu chce zapisac liste samych wynikow, nr chunka i ta nowa srednia, albo w kol 1 nowa srednia, w kol 2 srednia average header. w 3 std. w 4 srednia wszystkiego 
            #ale juz bez wartosci
        with open(filename + '_stats_only.csv', "w", newline='') as csv_file:
       
            result_1 = [['chunk'] + ['average for ' +  str(options.averages_for_x_ms['aver_len']) +' pts'] + ['head aver '+str(options.averages['head_len'])+' pts'] + ['tailaver '+str(options.averages['tail_len'])+' pts'] + ['h-t difference'] + [fun['label'] for fun in options.functions]] \
                       + [[row.number] + [row.average_for_x_ms] + [row.head_average] +[row.tail_average] +[row.difference] + row.functions for row in self.chunks if not row.dropped] \
                       + [['average'] + [''] + ['' for fun in options.functions]+['']]
                       
            # result_1 = [['chunk\\average'] +  [['averages fo 10 ms'] +[fun['label'] for fun in options.functions]] \
            #            + [[row.number] +[row.average_for_x_ms]+  row.functions for row in self.chunks if not row.dropped] \
            #            + [['average'] +[''] + ['' for fun in options.functions]]
            
            
            #     mean_only = [self.mean_for_x_ms]
        #     mean_onlyT = numpy.transpose(mean_only)
        
            writer = csv.writer(csv_file, delimiter = ',')
            writer.writerows(result_1)
            
            print('Finished processing file ' + str(fn))
            print('\n Saved options:')
            print('head length (pts): '+ str(options.averages['head_len']))
            print('tail length (pts): '+ str(options.averages['tail_len']))
            s= str(options.averages['tail_head_diff'])
            print('Head-tail difference abs value:' , s)
            #print('Arithmetic mean filters: ' + + 'to' + str(options.functions['filter_max']))
            print('Arithmetic mean filters: -100 to  100')

  
    def plot_all(self, filename):
        from matplotlib.ticker import AutoMinorLocator
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for chunk in self.chunks:
            if not chunk.dropped:
                ax1.plot(chunk.timestamps, chunk.values, linewidth='0.8')
                ax2.plot(chunk.timestamps, chunk.corrected_values, linewidth='0.8')
                
        ax1.plot(self.chunks[0].timestamps, self.timestamps_averages, 'r', label='averages')
        ax2.plot(self.chunks[0].timestamps, self.timestamps_averages_corrected, 'r', label='averages')
        ax1.set_ylabel('values')
        ax2.set_ylabel('values')
        ax1.set_xlabel('timestamps')
        ax2.set_xlabel('timestamps')
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.grid(True, which='major')
        ax2.grid(True, which='major')
        ax1.grid(True, which='minor', linestyle=':')
        ax2.grid(True, which='minor', linestyle=':')
        plt.title(str(fn)+' not shifted'+'mean 100 drops')
     #   plt.title(str(fn)+' all shifted '+'drops '+ 'h-t '+ str(options.averages['tail_head_diff']))
        
        fig1.savefig(filename + '_all.png')
        fig2.savefig(filename + '_all_corrected.png')
        

    def plot_averages(self, filename):
        from matplotlib.ticker import AutoMinorLocator
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.chunks[0].timestamps, self.timestamps_averages_corrected, 'r', label='average')
        ax.set_ylabel('values')
        ax.set_xlabel('timestamps')
        ax.legend(loc='best')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='major')
        ax.grid(True, which='minor', linestyle=':')
       # plt.title(str(fn) + 'with baseline shifted'+' drops ' + 'h-t '+ str(options.averages['tail_head_diff'])+'arith mean -0.05 to 0.15 ')
        plt.title(str(fn)+' not shifted'+'mean 100 drops')
        fig.savefig(filename + '_average.png')
        
    def plot_functions(self, filename):
         from matplotlib.ticker import AutoMinorLocator
        # import matplotlib.pyplot as plt

        # temp_chunks_no = [chunk.number for chunk in self.chunks if not chunk.dropped]
        # functions = [[chunk.functions_corrected[idx] for chunk in self.chunks if not chunk.dropped]
        #               for idx, function in enumerate(options.functions)]

        # fig, ax = plt.subplots(figsize=(10, 6))
        # for idx, function in enumerate(options.functions):
        #     ax.plot(temp_chunks_no, functions[idx], label=options.functions[idx]['label'])
        # ax.set_ylabel('values')
        # ax.set_xlabel('chunks')
        # ax.legend(loc='best')
        # ax.xaxis.set_minor_locator(AutoMinorLocator())
        # ax.yaxis.set_minor_locator(AutoMinorLocator())
        # ax.grid(True, which='major')
        # ax.grid(True, which='minor', linestyle=':')
        # fig.savefig(filename + '_functions.png')
        # plt.title('Mean corrected')

    def plot_functions_nc(self, filename):
        from matplotlib.ticker import AutoMinorLocator
        import matplotlib.pyplot as plt

        temp_chunks_no = [chunk.number for chunk in self.chunks if not chunk.dropped]
        functions_nc = [[chunk.functions[idx] for chunk in self.chunks if not chunk.dropped]
                      for idx, function in enumerate(options.functions)]

        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, function in enumerate(options.functions):
            ax.plot(temp_chunks_no, functions_nc[idx], label=options.functions[idx]['label'])
        ax.set_ylabel('values')
        ax.set_xlabel('chunks')
        ax.legend(loc='best')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='major')
        ax.grid(True, which='minor', linestyle=':')
        #plt.title(str(fn)+' dropped not corrected'' drops ' + 'h-t '+ str(options.averages['tail_head_diff'])+ 'arith mean -0.05 to 0.15')
        plt.title(str(fn)+' not shifted'+'no drops')
        fig.savefig(filename + '_functions_nc.png')
        

def read_files():
    import csv
    import glob
    
    file_list = sorted(glob.glob(options.input_directory + str(fn)+'.csv'))
    print('Files to read: ', file_list)

    chunk = Chunk()
    data = Data()

    for file in file_list:
        print('Opening file: %s' % file)
        f = open(file, newline='')
        reader = csv.reader(f, delimiter=options.input_delimiter)
        if options.headers:
            next(reader)

        for row in reader:
            chunk_no = int(row[0])
            timestamp = float(row[1])
            value = float(row[2])

            if not chunk.add_row(chunk_no, value, timestamp):
                data.add_chunk(chunk)
                chunk = Chunk()
                chunk.add_row(chunk_no, value, timestamp)

    data.add_chunk(chunk)
    return data


def main():
    import sys

    try:
        data = read_files()
        data.compute_chunks()
        data.save_to_file(options.output_name)
        data.plot_all(options.output_name)
        data.plot_averages(options.output_name)
        data.plot_functions(options.output_name)
        data.plot_functions_nc(options.output_name)
    except (IOError, OSError) as e:
        print(e)
        sys.exit(e.errno)
    except Exception as e:
        print(e)
        sys.exit(-1)


if __name__ == "__main__":
    options = Options()
    main()
