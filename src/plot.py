# module plot
"""Contains functions used for plotting."""

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray

import utils
from sim import Sim

if TYPE_CHECKING:
    from line import Line

PEN_WIDTH: int = 1


def plot_sample(
    plot_widget: pg.PlotWidget,
    wavenumbers: NDArray[np.float64],
    intensities: NDArray[np.float64],
    display_name: str,
) -> None:
    """Plot sample data.

    Args:
        plot_widget (pg.PlotWidget): A `GraphicsView` widget with a single `PlotItem` inside.
        wavenumbers (NDArray[np.float64]): Sample wavenumbers.
        intensities (NDArray[np.float64]): Sample intensities.
        display_name (str): The name of the file without directory information.
    """
    wavelengths: NDArray[np.float64] = utils.wavenum_to_wavelen(wavenumbers)

    plot_widget.plot(
        wavelengths,
        intensities / intensities.max(),
        pen=pg.mkPen("w", width=PEN_WIDTH),
        name=display_name,
    )


def plot_line(plot_widget: pg.PlotWidget, sim: Sim, colors: list[str]) -> None:
    """Plot each rotational line.

    Args:
        plot_widget (pg.PlotWidget): A `GraphicsView` widget with a single `PlotItem` inside.
        sim (Sim): The parent simulation.
        colors (list[str]): A list of colors for plotting.
    """
    max_intensity: float = sim.all_line_data()[1].max()

    for idx, band in enumerate(sim.bands):
        wavelengths_line: NDArray[np.float64] = utils.wavenum_to_wavelen(band.wavenumbers_line())
        intensities_line: NDArray[np.float64] = band.intensities_line()

        # Create a scatter plot with points at zero and peak intensity.
        scatter_data: NDArray[np.float64] = np.column_stack(
            [
                np.repeat(wavelengths_line, 2),
                np.column_stack(
                    [np.zeros_like(wavelengths_line), intensities_line / max_intensity]
                ).flatten(),
            ],
        ).astype(np.float64)

        plot_widget.plot(
            scatter_data[:, 0],
            scatter_data[:, 1],
            pen=pg.mkPen(colors[idx], width=PEN_WIDTH),
            connect="pairs",
            name=f"{sim.molecule.name} {band.v_qn_up, band.v_qn_lo} line",
        )


def plot_line_info(plot_widget: pg.PlotWidget, sim: Sim, colors: list[str]) -> None:
    """Plot information about each rotational line.

    Args:
        plot_widget (pg.PlotWidget): A `GraphicsView` widget with a single `PlotItem` inside.
        sim (Sim): The parent simulation.
        colors (list[str]): A list of colors for plotting.
    """
    # In order to show text, a plot must first exist.
    plot_line(plot_widget, sim, colors)

    max_intensity: float = sim.all_line_data()[1].max()

    for band in sim.bands:
        # Only select non-satellite lines to reduce the amount of data on screen.
        lines: list[Line] = [line for line in band.lines if not line.is_satellite]

        for line in lines:
            wavelength: float = utils.wavenum_to_wavelen(line.wavenumber)
            intensity: float = line.intensity / max_intensity
            text: pg.TextItem = pg.TextItem(
                f"{line.branch_name}_{line.branch_idx_up}{line.branch_idx_lo}",
                color="w",
                anchor=(0.5, 1.2),
            )
            plot_widget.addItem(text)
            text.setPos(wavelength, intensity)


def plot_conv_sep(
    plot_widget: pg.PlotWidget,
    sim: Sim,
    colors: list[str],
    fwhm_selections: dict[str, bool],
    inst_broadening_wl: float,
    granularity: int,
) -> None:
    """Plot convolved data for each vibrational band separately.

    Args:
        plot_widget (pg.PlotWidget): A `GraphicsView` widget with a single `PlotItem` inside.
        sim (Sim): The parent simulation.
        colors (list[str]): A list of colors for plotting.
        fwhm_selections (dict[str, bool]): The types of broadening to be simulated.
        inst_broadening_wl (float): Instrument broadening FWHM in [nm].
        granularity (int): Number of points on the wavenumber axis.
    """
    # Need to convolve all bands separately, get their maximum intensities, store the largest, and
    # then divide all bands by that maximum. If the max intensity was found for all bands convolved
    # together, it would be inaccurate because of vibrational band overlap.
    max_intensity: float = max(
        band.intensities_conv(
            fwhm_selections,
            inst_broadening_wl,
            band.wavenumbers_conv(inst_broadening_wl, granularity),
        ).max()
        for band in sim.bands
    )

    for idx, band in enumerate(sim.bands):
        wavelengths_conv: NDArray[np.float64] = utils.wavenum_to_wavelen(
            band.wavenumbers_conv(inst_broadening_wl, granularity)
        )
        intensities_conv: NDArray[np.float64] = band.intensities_conv(
            fwhm_selections,
            inst_broadening_wl,
            band.wavenumbers_conv(inst_broadening_wl, granularity),
        )

        plot_widget.plot(
            wavelengths_conv,
            intensities_conv / max_intensity,
            pen=pg.mkPen(colors[idx], width=PEN_WIDTH),
            name=f"{sim.molecule.name} {band.v_qn_up, band.v_qn_lo} conv",
        )


def plot_conv_all(
    plot_widget: pg.PlotWidget,
    sim: Sim,
    colors: list[str],
    fwhm_selections: dict[str, bool],
    inst_broadening_wl: float,
    granularity: int,
) -> None:
    """Plot convolved data for all vibrational bands simultaneously.

    Args:
        plot_widget (pg.PlotWidget): A `GraphicsView` widget with a single `PlotItem` inside.
        sim (Sim): The parent simulation.
        colors (list[str]): A list of colors for plotting.
        fwhm_selections (dict[str, bool]): The types of broadening to be simulated.
        inst_broadening_wl (float): Instrument broadening FWHM in [nm].
        granularity (int): Number of points on the wavenumber axis.
    """
    wavenumbers_conv, intensities_conv = sim.all_conv_data(
        fwhm_selections, inst_broadening_wl, granularity
    )
    wavelengths_conv: NDArray[np.float64] = utils.wavenum_to_wavelen(wavenumbers_conv)

    plot_widget.plot(
        wavelengths_conv,
        intensities_conv / intensities_conv.max(),
        pen=pg.mkPen(colors[0], width=PEN_WIDTH),
        name=f"{sim.molecule.name} conv all",
    )
