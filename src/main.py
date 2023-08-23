# module main
'''
Computes spectral lines for triplet oxygen.
'''

import numpy as np

import output as out
import input as inp
import bands

def main():
    '''
    Runs the program.
    '''

    # Temperature used in Cosby is 300 K
    # Pressure used in Cosby is 20 Torr (2666.45 Pa)
    # pres = 2666.45
    # v_00 = 36185

    # Read the table of Franck-Condon factors into a 2-D numpy array
    fc_data = np.loadtxt('../data/test.csv', delimiter=',')

    # Create a vibrational band line plot for each of the user-selected bands
    band_list = []
    for band in inp.VIB_BANDS:
        band_list.append(bands.LinePlot(inp.TEMP, inp.PRES, inp.ROT_LVLS, band))

    # Find the maximum Franck-Condon factor of all the bands, this is used to normalize the
    # intensities of each band with respect to the largest band
    max_fc = max((band.get_fc(fc_data) for band in band_list))

    out.plot_style()

    if inp.LINE_DATA:
        line_data   = [band.get_line(fc_data, max_fc) for band in band_list]
        line_colors = ['black', 'red']
        line_labels = [str(band) + ' Band' for band in inp.VIB_BANDS]

        out.plot_line(line_data, line_colors, line_labels)

    if inp.CONV_DATA:
        conv_data   = [band.get_conv(fc_data, max_fc) for band in band_list]
        conv_colors = ['blue', 'green']
        conv_labels = ['Convolved ' + str(band) + ' Band' for band in inp.VIB_BANDS]

        out.plot_conv(conv_data, conv_colors, conv_labels)

    if inp.SAMP_DATA:
        samp_files  = ['harvard', 'pgopher']
        samp_data   = []
        for file in samp_files:
            samp_data.append(out.configure_samples(file))
        samp_colors = ['pink', 'orange']
        samp_labels = ['Harvard Data', 'PGOPHER Data']

        out.plot_samp(samp_data, samp_colors, samp_labels)

    out.show_plot()

if __name__ == '__main__':
    main()
