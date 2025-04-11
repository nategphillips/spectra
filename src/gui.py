# module gui
"""A GUI built using PySide6 with a native table view for DataFrames."""

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPoint, QRect, Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import plot
import utils
from atom import Atom
from colors import get_colors
from molecule import Molecule
from sim import Sim
from simtype import SimType
from state import State

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

DEFAULT_LINES: int = 40
DEFAULT_GRANULARITY: int = int(1e4)

DEFAULT_TEMPERATURE: float = 300.0  # [K]
DEFAULT_PRESSURE: float = 101325.0  # [Pa]
DEFAULT_BROADENING: float = 0.0  # [nm]

DEFAULT_BANDS: str = "0-0"
DEFAULT_PLOTTYPE: str = "Line"
DEFAULT_SIMTYPE: str = "Absorption"


class MyDoubleSpinBox(QDoubleSpinBox):
    """A custom double spin box.

    Allows for arbitrarily large or small input values, high decimal precision, and scientific
    notation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(-1e99, 1e99)
        self.setDecimals(6)
        self.setKeyboardTracking(False)

    def valueFromText(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return 0.0

    def textFromValue(self, value: float) -> str:
        return f"{value:g}"

    def validate(self, text: str, pos: int):
        # Allow empty input.
        if text == "":
            return (QValidator.State.Intermediate, text, pos)
        try:
            # Try converting to float.
            float(text)
            return (QValidator.State.Acceptable, text, pos)
        except ValueError:
            # If the text contains an 'e' or 'E', it might be a partial scientific notation.
            if "e" in text.lower():
                parts = text.lower().split("e")
                # Allow cases like "1e", "1e-", or "1e+".
                if len(parts) == 2 and (parts[1] == "" or parts[1] in ["-", "+"]):
                    return (QValidator.State.Intermediate, text, pos)
            return (QValidator.State.Invalid, text, pos)


class MyTable(QAbstractTableModel):
    """A simple model to interface a Qt view with a DataFrame."""

    def __init__(self, df: pl.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=QModelIndex()):
        return self._df.height

    def columnCount(self, parent=QModelIndex()):
        return self._df.width

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._df[index.row(), index.column()]
            column_name = self._df.columns[index.column()]

            # NOTE: 25/04/10 - This only changes the values displayed to the user using the built-in
            #       table view. If the table is exported, the underlying dataframe is used instead,
            #       which retains the full-precision values calculated by the simulation.
            if isinstance(value, float):
                if "Intensity" in column_name:
                    return f"{value:.4e}"
                return f"{value:.4f}"

            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        if orientation == Qt.Orientation.Vertical:
            return str(section)
        return None


def create_dataframe_tab(df: pl.DataFrame, tab_label: str) -> QWidget:
    """Create a QWidget containing a QTableView to display the DataFrame."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    table_view = QTableView()
    model = MyTable(df)
    table_view.setModel(model)

    # TODO: 25/04/10 - Enabling column resizing dramatically increases the time it takes to render
    #       tables with even a moderate number of bands. Keeping this disabled unless there's a
    #       faster way to achieve the same thing.
    # table_view.resizeColumnsToContents()

    layout.addWidget(table_view)

    return widget


class GUI(QMainWindow):
    """The GUI implemented with PySide6."""

    def __init__(self) -> None:
        super().__init__()

        pg.setConfigOption("background", (30, 30, 30))
        pg.setConfigOption("foreground", "w")

        # NOTE: 25/03/27 - Enabling antialiasing with a line width greater than 1 leads to severe
        #       performance issues. Setting `useOpenGL=True` doesn't help since antialiasing does
        #       not work with OpenGL from what I've seen. Relevant topics:
        #       - https://github.com/pyqtgraph/pyqtgraph/issues/533
        #       - https://pyqtgraph.narkive.com/aIpWRh9F/is-antialiasing-for-2d-plots-with-opengl-not-supported
        #
        # pg.setConfigOptions(antialias=True, useOpenGL=True)

        self.setWindowTitle("Diatomic Molecular Simulation")
        self.resize(1600, 800)
        self.center()
        self.init_ui()

    def center(self) -> None:
        """Center the window on the screen."""
        qr: QRect = self.frameGeometry()
        qp: QPoint = self.screen().availableGeometry().center()
        qr.moveCenter(qp)
        self.move(qr.topLeft())

    def init_ui(self) -> None:
        """Initialize the user interface."""
        central_widget: QWidget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout: QVBoxLayout = QVBoxLayout(central_widget)

        # Top panel (above)
        top_panel: QWidget = self.create_top_panel()
        main_layout.addWidget(top_panel)

        # Main panel (table and plot)
        main_panel: QWidget = self.create_main_panel()
        main_layout.addWidget(main_panel, stretch=1)

        # Bottom panel (input entries and combos)
        bottom_panel: QWidget = self.create_bottom_panel()
        main_layout.addWidget(bottom_panel)

    def create_top_panel(self) -> QWidget:
        """Create the top panel with band ranges, broadening, granularity, and run controls."""
        top_widget: QWidget = QWidget()
        layout: QHBoxLayout = QHBoxLayout(top_widget)

        # --- Group 1: Bands and broadening checkboxes ---
        group_bands: QGroupBox = QGroupBox("Bands")
        bands_layout: QVBoxLayout = QVBoxLayout(group_bands)

        # TODO: 25/04/09 - Changing the mode from specific bands to band ranges resizes the UI,
        #       which is really annoying. Fix this in the future.

        # Add radio buttons for selection mode
        band_selection_layout = QHBoxLayout()
        self.radio_specific_bands = QRadioButton("Specific Bands")
        self.radio_specific_bands.setChecked(True)
        self.radio_band_ranges = QRadioButton("Band Ranges")
        band_selection_layout.addWidget(self.radio_specific_bands)
        band_selection_layout.addWidget(self.radio_band_ranges)
        bands_layout.addLayout(band_selection_layout)

        # Connect radio buttons to toggle band input method
        self.radio_specific_bands.toggled.connect(self.toggle_band_input_method)

        # Create container for specific bands input
        self.specific_bands_container = QWidget()
        specific_bands_layout = QVBoxLayout(self.specific_bands_container)
        specific_bands_layout.setContentsMargins(0, 0, 0, 0)

        # Row: Band ranges entry (specific bands method)
        band_range_layout = QHBoxLayout()
        band_range_label = QLabel("Bands:")
        self.band_ranges_line_edit = QLineEdit(DEFAULT_BANDS)
        band_range_layout.addWidget(band_range_label)
        band_range_layout.addWidget(self.band_ranges_line_edit)
        specific_bands_layout.addLayout(band_range_layout)

        # Create container for band ranges input
        self.band_ranges_container = QWidget()
        band_ranges_layout = QGridLayout(self.band_ranges_container)
        band_ranges_layout.setContentsMargins(0, 0, 0, 0)

        # Upper state vibrational levels
        v_up_min_label = QLabel("v' min:")
        self.v_up_min_spinbox = QSpinBox()
        self.v_up_min_spinbox.setRange(0, 99)
        self.v_up_min_spinbox.setValue(0)

        v_up_max_label = QLabel("v' max:")
        self.v_up_max_spinbox = QSpinBox()
        self.v_up_max_spinbox.setRange(0, 99)
        self.v_up_max_spinbox.setValue(10)

        # Lower state vibrational levels
        v_lo_min_label = QLabel("v'' min:")
        self.v_lo_min_spinbox = QSpinBox()
        self.v_lo_min_spinbox.setRange(0, 99)
        self.v_lo_min_spinbox.setValue(5)

        v_lo_max_label = QLabel("v'' max:")
        self.v_lo_max_spinbox = QSpinBox()
        self.v_lo_max_spinbox.setRange(0, 99)
        self.v_lo_max_spinbox.setValue(5)

        # Add upper state widgets to grid
        band_ranges_layout.addWidget(v_up_min_label, 0, 0)
        band_ranges_layout.addWidget(self.v_up_min_spinbox, 0, 1)
        band_ranges_layout.addWidget(v_up_max_label, 0, 2)
        band_ranges_layout.addWidget(self.v_up_max_spinbox, 0, 3)

        # Add lower state widgets to grid
        band_ranges_layout.addWidget(v_lo_min_label, 1, 0)
        band_ranges_layout.addWidget(self.v_lo_min_spinbox, 1, 1)
        band_ranges_layout.addWidget(v_lo_max_label, 1, 2)
        band_ranges_layout.addWidget(self.v_lo_max_spinbox, 1, 3)

        # Initial setup - hide band ranges, show specific bands
        self.band_ranges_container.hide()

        # Add both containers to the bands layout
        bands_layout.addWidget(self.specific_bands_container)
        bands_layout.addWidget(self.band_ranges_container)

        layout.addWidget(group_bands)

        # --- Group 2: Instrument Broadening value ---
        group_inst_broadening = QGroupBox("Instrument Broadening [nm]")
        inst_layout = QHBoxLayout(group_inst_broadening)
        self.inst_broadening_spinbox = MyDoubleSpinBox()
        self.inst_broadening_spinbox.setValue(DEFAULT_BROADENING)
        inst_layout.addWidget(self.inst_broadening_spinbox)

        # Row: Broadening checkboxes.
        checkbox_layout = QHBoxLayout()
        self.checkbox_instrument = QCheckBox("Instrument")
        self.checkbox_instrument.setChecked(True)
        self.checkbox_doppler = QCheckBox("Doppler")
        self.checkbox_doppler.setChecked(True)
        self.checkbox_natural = QCheckBox("Natural")
        self.checkbox_natural.setChecked(True)
        self.checkbox_collisional = QCheckBox("Collisional")
        self.checkbox_collisional.setChecked(True)
        self.checkbox_predissociation = QCheckBox("Predissociation")
        self.checkbox_predissociation.setChecked(True)
        checkbox_layout.addWidget(self.checkbox_instrument)
        checkbox_layout.addWidget(self.checkbox_doppler)
        checkbox_layout.addWidget(self.checkbox_natural)
        checkbox_layout.addWidget(self.checkbox_collisional)
        checkbox_layout.addWidget(self.checkbox_predissociation)
        inst_layout.addLayout(checkbox_layout)
        layout.addWidget(group_inst_broadening)

        # --- Group 3: Granularity ---
        group_granularity = QGroupBox("Granularity")
        gran_layout = QHBoxLayout(group_granularity)
        self.granularity_spinbox = QSpinBox()
        self.granularity_spinbox.setMaximum(10000000)
        self.granularity_spinbox.setValue(DEFAULT_GRANULARITY)
        gran_layout.addWidget(self.granularity_spinbox)
        layout.addWidget(group_granularity)

        # --- Group 4: Rotational lines ---
        group_lines = QGroupBox("Rotational Lines")
        lines_layout = QHBoxLayout(group_lines)
        self.num_lines_spinbox = QSpinBox()
        self.num_lines_spinbox.setMaximum(10000)
        self.num_lines_spinbox.setValue(DEFAULT_LINES)
        lines_layout.addWidget(self.num_lines_spinbox)
        layout.addWidget(group_lines)

        # --- Group 5: Action buttons ---
        group_run = QGroupBox("Actions")
        run_layout = QHBoxLayout(group_run)
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.add_simulation)
        run_layout.addWidget(self.run_button)
        self.open_button = QPushButton("Open File")
        self.open_button.clicked.connect(self.add_sample)
        run_layout.addWidget(self.open_button)
        self.export_button = QPushButton("Export Table")
        self.export_button.clicked.connect(self.export_current_table)
        run_layout.addWidget(self.export_button)
        layout.addWidget(group_run)

        return top_widget

    def toggle_band_input_method(self, checked: bool) -> None:
        """Toggle between band input methods based on radio button selection.

        Args:
            checked (bool): True if specific bands radio button is checked
        """
        if checked:
            self.specific_bands_container.show()
            self.band_ranges_container.hide()
        else:
            self.specific_bands_container.hide()
            self.band_ranges_container.show()

    def create_main_panel(self) -> QWidget:
        """Create the main panel with table tabs on the left and a plot on the right."""
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)

        # --- Left side: QTabWidget for tables ---
        self.tab_widget = QTabWidget()
        # Add an initial empty tab (using an empty DataFrame)
        empty_df = pl.DataFrame()
        empty_tab = create_dataframe_tab(empty_df, "v'-v''")
        self.tab_widget.addTab(empty_tab, "v'-v''")
        layout.addWidget(self.tab_widget, stretch=1)

        # --- Right side: Plot area ---
        self.plot_widget = pg.PlotWidget()

        # Adds a legend at the top right of the plot.
        self.plot_widget.addLegend(offset=(0, 1))
        self.plot_widget.setAxisItems({"top": WavenumberAxis(orientation="top")})
        self.plot_widget.setLabel("top", "Wavenumber, ν [cm<sup>-1</sup>]")
        self.plot_widget.setLabel("bottom", "Wavelength, λ [nm]")
        self.plot_widget.setLabel("left", "Intensity, I [a.u.]")
        self.plot_widget.setLabel("right", "Intensity, I [a.u.]")

        # Initial arbitrary wavelength range
        self.plot_widget.setXRange(100, 200)

        layout.addWidget(self.plot_widget, stretch=2)
        self.legend = self.plot_widget.addLegend()

        return main_widget

    def create_bottom_panel(self) -> QWidget:
        """Create the bottom panel with temperature, pressure, and combo selections."""
        bottom_widget = QWidget()
        layout = QHBoxLayout(bottom_widget)

        # --- Left: Input entries for temperature and pressure ---
        entries_widget = QWidget()
        entries_layout = QGridLayout(entries_widget)

        # Row 0: Equilibrium temperature.
        self.temp_label = QLabel("Temperature [K]:")
        self.temp_spinbox = MyDoubleSpinBox()
        self.temp_spinbox.setValue(DEFAULT_TEMPERATURE)
        entries_layout.addWidget(self.temp_label, 0, 0)
        entries_layout.addWidget(self.temp_spinbox, 0, 1)

        # Nonequilibrium temperatures (hidden initially)
        self.temp_trn_label = QLabel("Translational Temp [K]:")
        self.temp_trn_spinbox = MyDoubleSpinBox()
        self.temp_trn_spinbox.setValue(DEFAULT_TEMPERATURE)
        self.temp_elc_label = QLabel("Electronic Temp [K]:")
        self.temp_elc_spinbox = MyDoubleSpinBox()
        self.temp_elc_spinbox.setValue(DEFAULT_TEMPERATURE)
        self.temp_vib_label = QLabel("Vibrational Temp [K]:")
        self.temp_vib_spinbox = MyDoubleSpinBox()
        self.temp_vib_spinbox.setValue(DEFAULT_TEMPERATURE)
        self.temp_rot_label = QLabel("Rotational Temp [K]:")
        self.temp_rot_spinbox = MyDoubleSpinBox()
        self.temp_rot_spinbox.setValue(DEFAULT_TEMPERATURE)

        for w in (
            self.temp_trn_label,
            self.temp_trn_spinbox,
            self.temp_elc_label,
            self.temp_elc_spinbox,
            self.temp_vib_label,
            self.temp_vib_spinbox,
            self.temp_rot_label,
            self.temp_rot_spinbox,
        ):
            w.hide()

        entries_layout.addWidget(self.temp_trn_label, 0, 2)
        entries_layout.addWidget(self.temp_trn_spinbox, 0, 3)
        entries_layout.addWidget(self.temp_elc_label, 0, 4)
        entries_layout.addWidget(self.temp_elc_spinbox, 0, 5)
        entries_layout.addWidget(self.temp_vib_label, 0, 6)
        entries_layout.addWidget(self.temp_vib_spinbox, 0, 7)
        entries_layout.addWidget(self.temp_rot_label, 0, 8)
        entries_layout.addWidget(self.temp_rot_spinbox, 0, 9)

        # Row 1: Pressure.
        pressure_label = QLabel("Pressure [Pa]:")
        self.pressure_spinbox = MyDoubleSpinBox()
        self.pressure_spinbox.setValue(DEFAULT_PRESSURE)
        entries_layout.addWidget(pressure_label, 1, 0)
        entries_layout.addWidget(self.pressure_spinbox, 1, 1)

        layout.addWidget(entries_widget)

        # --- Right: Combo boxes ---
        combos_widget = QWidget()
        combos_layout = QGridLayout(combos_widget)

        # Temperature Type.
        temp_type_label = QLabel("Temperature Type:")
        self.temp_type_combo = QComboBox()
        self.temp_type_combo.addItems(["Equilibrium", "Nonequilibrium"])
        self.temp_type_combo.currentTextChanged.connect(self.switch_temp_mode)
        combos_layout.addWidget(temp_type_label, 0, 0)
        combos_layout.addWidget(self.temp_type_combo, 0, 1)

        # Simulation Type.
        sim_type_label = QLabel("Simulation Type:")
        self.sim_type_combo = QComboBox()
        self.sim_type_combo.addItems(["Absorption", "Emission"])
        combos_layout.addWidget(sim_type_label, 1, 0)
        combos_layout.addWidget(self.sim_type_combo, 1, 1)

        # Plot Type.
        plot_type_label = QLabel("Plot Type:")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Line", "Line Info", "Convolve Separate", "Convolve All"])
        combos_layout.addWidget(plot_type_label, 2, 0)
        combos_layout.addWidget(self.plot_type_combo, 2, 1)

        layout.addWidget(combos_widget)

        return bottom_widget

    def add_sample(self) -> None:
        """Open a CSV file and adds a new tab showing its contents."""
        filename, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open File",
            dir=str(utils.get_data_path("data", "samples")),
            filter="CSV Files (*.csv);;All Files (*)",
        )
        if filename:
            try:
                df: pl.DataFrame = pl.read_csv(filename)
            except ValueError:
                QMessageBox.critical(self, "Error", "Data is improperly formatted.")
                return
        else:
            return

        display_name: str = Path(filename).name

        new_tab: QWidget = create_dataframe_tab(df, display_name)
        self.tab_widget.addTab(new_tab, display_name)

        wavenumbers: NDArray[np.float64] = df["wavenumber"].to_numpy()
        intensities: NDArray[np.float64] = df["intensity"].to_numpy()

        plot.plot_sample(self.plot_widget, wavenumbers, intensities, display_name)

        # Update the wavelength range automatically based on the plotted data.
        self.plot_widget.autoRange()

    def switch_temp_mode(self) -> None:
        """Switch between equilibrium and nonequilibrium temperature modes."""
        if self.temp_type_combo.currentText() == "Nonequilibrium":
            self.temp_label.hide()
            self.temp_spinbox.hide()
            self.temp_trn_label.show()
            self.temp_trn_spinbox.show()
            self.temp_elc_label.show()
            self.temp_elc_spinbox.show()
            self.temp_vib_label.show()
            self.temp_vib_spinbox.show()
            self.temp_rot_label.show()
            self.temp_rot_spinbox.show()
        else:
            self.temp_trn_label.hide()
            self.temp_trn_spinbox.hide()
            self.temp_elc_label.hide()
            self.temp_elc_spinbox.hide()
            self.temp_vib_label.hide()
            self.temp_vib_spinbox.hide()
            self.temp_rot_label.hide()
            self.temp_rot_spinbox.hide()
            self.temp_label.show()
            self.temp_spinbox.show()

    def parse_band_ranges(self) -> list[tuple[int, int]]:
        """Parse comma-separated band ranges from user input."""
        band_ranges_str: str = self.band_ranges_line_edit.text()
        bands: list[tuple[int, int]] = []

        for range_str in band_ranges_str.split(","):
            if "-" in range_str.strip():
                try:
                    v_up, v_lo = map(int, range_str.split("-"))
                    bands.append((v_up, v_lo))
                except ValueError:
                    QMessageBox.information(
                        self,
                        "Info",
                        f"Invalid band range format: {range_str}",
                        QMessageBox.StandardButton.Ok,
                    )
            else:
                QMessageBox.information(
                    self,
                    "Info",
                    f"Invalid band range format: {range_str}",
                    QMessageBox.StandardButton.Ok,
                )

        return bands

    def add_simulation(self) -> None:
        """Run a simulation instance and update the plot and table tabs."""
        start_time: float = time.time()

        # Determine temperatures based on mode.
        temp: float = self.temp_spinbox.value()
        temp_trn = temp_elc = temp_vib = temp_rot = temp
        if self.temp_type_combo.currentText() == "Nonequilibrium":
            temp_trn = self.temp_trn_spinbox.value()
            temp_elc = self.temp_elc_spinbox.value()
            temp_vib = self.temp_vib_spinbox.value()
            temp_rot = self.temp_rot_spinbox.value()

        pres: float = self.pressure_spinbox.value()
        sim_type: SimType = SimType[self.sim_type_combo.currentText().upper()]

        # Get bands based on the selected method
        if self.radio_specific_bands.isChecked():
            # Use the existing parse_band_ranges method
            bands: list[tuple[int, int]] = self.parse_band_ranges()
            if not bands:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "No valid band ranges specified. Please check your input.",
                    QMessageBox.StandardButton.Ok,
                )
                return
        else:
            # Use the new band range spinboxes
            v_up_min: int = self.v_up_min_spinbox.value()
            v_up_max: int = self.v_up_max_spinbox.value()
            v_lo_min: int = self.v_lo_min_spinbox.value()
            v_lo_max: int = self.v_lo_max_spinbox.value()

            # Validate input
            if v_up_min > v_up_max or v_lo_min > v_lo_max:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Invalid band range: min value cannot be greater than max value.",
                    QMessageBox.StandardButton.Ok,
                )
                return

            # Generate band combinations based on the ranges
            if (v_up_min == v_up_max) and (v_lo_min == v_lo_max):
                bands = [(v_up_min, v_lo_min)]
            elif v_up_min == v_up_max:
                bands = [(v_up_min, v_lo) for v_lo in range(v_lo_min, v_lo_max + 1)]
            elif v_lo_min == v_lo_max:
                bands = [(v_up, v_lo_min) for v_up in range(v_up_min, v_up_max + 1)]
            else:
                bands = [
                    (v_up, v_lo)
                    for v_up in range(v_up_min, v_up_max + 1)
                    for v_lo in range(v_lo_min, v_lo_max + 1)
                ]

        rot_lvls = np.arange(0, self.num_lines_spinbox.value())

        molecule: Molecule = Molecule(name="O2", atom_1=Atom("O"), atom_2=Atom("O"))
        state_up: State = State(name="B3Su-", spin_multiplicity=3, molecule=molecule)
        state_lo: State = State(name="X3Sg-", spin_multiplicity=3, molecule=molecule)

        sim: Sim = Sim(
            sim_type=sim_type,
            molecule=molecule,
            state_up=state_up,
            state_lo=state_lo,
            rot_lvls=rot_lvls,
            temp_trn=temp_trn,
            temp_elc=temp_elc,
            temp_vib=temp_vib,
            temp_rot=temp_rot,
            pressure=pres,
            bands=bands,
        )

        print(f"Time to create sim: {time.time() - start_time} s")
        start_plot_time: float = time.time()

        colors: list[str] = get_colors(bands)

        self.plot_widget.clear()

        # Map plot types to functions.
        map_functions: dict[str, Callable] = {
            "Line": plot.plot_line,
            "Line Info": plot.plot_line_info,
            "Convolve Separate": plot.plot_conv_sep,
            "Convolve All": plot.plot_conv_all,
        }
        plot_type: str = self.plot_type_combo.currentText()
        plot_function: Callable | None = map_functions.get(plot_type)

        # Check which FWHM parameters the user has selected.
        fwhm_selections: dict[str, bool] = {
            "instrument": self.checkbox_instrument.isChecked(),
            "doppler": self.checkbox_doppler.isChecked(),
            "natural": self.checkbox_natural.isChecked(),
            "collisional": self.checkbox_collisional.isChecked(),
            "predissociation": self.checkbox_predissociation.isChecked(),
        }

        if plot_function is not None:
            if plot_function.__name__ in ("plot_conv_sep", "plot_conv_all"):
                plot_function(
                    self.plot_widget,
                    sim,
                    colors,
                    fwhm_selections,
                    self.inst_broadening_spinbox.value(),
                    self.granularity_spinbox.value(),
                )
            else:
                plot_function(self.plot_widget, sim, colors)
        else:
            QMessageBox.information(
                self,
                "Info",
                f"Plot type '{plot_type}' is not recognized.",
                QMessageBox.StandardButton.Ok,
            )

        # Update the wavelength range automatically based on the plotted data.
        self.plot_widget.autoRange()

        print(f"Time to create plot: {time.time() - start_plot_time} s")
        start_table_time: float = time.time()

        # Clear previous tabs.
        while self.tab_widget.count() > 0:
            self.tab_widget.removeTab(0)

        # Create a new tab for each vibrational band.
        for i, band in enumerate(bands):
            df: pl.DataFrame = pl.DataFrame(
                [
                    {
                        "Wavelength": utils.wavenum_to_wavelen(line.wavenumber),
                        "Wavenumber": line.wavenumber,
                        "Intensity": line.intensity,
                        "J'": line.j_qn_up,
                        "J''": line.j_qn_lo,
                        "N'": line.n_qn_up,
                        "N''": line.n_qn_lo,
                        "Branch": f"{line.branch_name}{line.branch_idx_up}{line.branch_idx_lo}",
                    }
                    for line in sim.bands[i].lines
                ]
            )

            tab_name: str = f"{band[0]}-{band[1]}"
            new_tab: QWidget = create_dataframe_tab(df, tab_name)
            self.tab_widget.addTab(new_tab, tab_name)

        print(f"Time to create table: {time.time() - start_table_time} s")
        print(f"Total time: {time.time() - start_time} s\n")

    def export_current_table(self) -> None:
        """Export the currently displayed table to a CSV file."""
        current_widget: QWidget = self.tab_widget.currentWidget()

        # TODO: 25/03/28 - Currently only the currently selected table is grabbed and exported. In
        #       the future, there should be an option to export any number of tables.
        table_view = current_widget.findChild(QTableView)
        if table_view is None:
            QMessageBox.information(
                self,
                "Error",
                "No table view found in this tab.",
                QMessageBox.StandardButton.Ok,
            )
            return

        model: MyTable = table_view.model()
        if not hasattr(model, "_df"):
            QMessageBox.information(
                self,
                "Error",
                "This table does not support CSV export.",
                QMessageBox.StandardButton.Ok,
            )
            return

        df: pl.DataFrame = model._df

        filename, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if filename:
            try:
                df.write_csv(filename)
                QMessageBox.information(
                    self, "Success", "Table exported successfully.", QMessageBox.StandardButton.Ok
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"An error occurred: {e}")


class WavenumberAxis(pg.AxisItem):
    """A custom x-axis displaying wavenumbers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, wavelengths: list[float], *_) -> list[str]:
        """Return the wavenumber strings that are placed next to ticks.

        Args:
            wavelengths (list[float]): List of wavelength values.

        Returns:
            list[str]: List of wavenumber values placed next to ticks.
        """
        strings: list[str] = []

        for wavelength in wavelengths:
            if wavelength != 0:
                wavenumber: float = utils.wavenum_to_wavelen(wavelength)
                strings.append(f"{wavenumber:.1f}")
            else:
                strings.append("∞")

        return strings


def main() -> None:
    """Entry point."""
    app: QApplication = QApplication(sys.argv)
    window: GUI = GUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
