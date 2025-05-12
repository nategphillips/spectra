# module main.py
"""A simulation of the Schumann-Runge bands of molecular oxygen written in Python."""

# Copyright (C) 2025 Nathan Phillips

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPoint,
    QRect,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPaintEvent, QPen, QValidator
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

    def __init__(self) -> None:
        """Initialize class variables."""
        super().__init__()
        self.setRange(-1e99, 1e99)
        self.setDecimals(6)
        self.setKeyboardTracking(False)

    def valueFromText(self, text: str) -> float:  # noqa: N802
        """Reads text from the input field and converts it to a float.

        Args:
            text (str): Input text.

        Returns:
            float: Converted value.
        """
        try:
            return float(text)
        except ValueError:
            return 0.0

    def textFromValue(self, value: float) -> str:  # noqa: N802
        """Displays the input parameter using f-strings.

        Args:
            value (float): Input float.

        Returns:
            str: Displayed text.
        """
        return f"{value:g}"

    def validate(self, text: str, pos: int) -> tuple[QValidator.State, str, int]:
        """Checks user input and returns the correct state.

        Args:
            text (str): Input text.
            pos (int): Position in the string.

        Returns:
            tuple[QValidator.State, str, int]: The current state, input text, and string position.
        """
        if text == "":
            return (QValidator.State.Intermediate, text, pos)
        try:
            float(text)
            return (QValidator.State.Acceptable, text, pos)
        except ValueError:
            # If the string cannot immediately be converted to a float, first check for scientific
            # notation and return an intermediate state if found.
            if "e" in text.lower():
                parts: list[str] = text.lower().split("e")
                num_parts: int = 2
                if len(parts) == num_parts and (parts[1] == "" or parts[1] in ["-", "+"]):
                    return (QValidator.State.Intermediate, text, pos)
            return (QValidator.State.Invalid, text, pos)


class MyTable(QAbstractTableModel):
    """A simple model to interface a Qt view with a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame) -> None:
        """Initialize class variables.

        Args:
            df (pl.DataFrame): A Polars `DataFrame`.
        """
        super().__init__()
        self.df: pl.DataFrame = df

    def rowCount(self, _: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        """Get the height of the table.

        Args:
            _ (QModelIndex, optional): Parent class used to locate data. Defaults to QModelIndex().

        Returns:
            int: Table height.
        """
        return self.df.height

    def columnCount(self, _: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        """Get the width of the table.

        Args:
            _ (QModelIndex, optional): Parent class used to locate data. Defaults to QModelIndex().

        Returns:
            int: Table width.
        """
        return self.df.width

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> str | None:
        """Validates and formats data displayed in the table.

        Args:
            index (QModelIndex): Parent class used to locate data.
            role (int, optional): Data rendered as text. Defaults to Qt.ItemDataRole.DisplayRole.

        Returns:
            str | None: The formatted text.
        """
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value: int | float | str = self.df[index.row(), index.column()]
            column_name: str = self.df.columns[index.column()]

            # NOTE: 25/04/10 - This only changes the values displayed to the user using the built-in
            #       table view. If the table is exported, the underlying dataframe is used instead,
            #       which retains the full-precision values calculated by the simulation.
            if isinstance(value, float):
                if "Intensity" in column_name:
                    return f"{value:.4e}"
                return f"{value:.4f}"

            return str(value)
        return None

    def headerData(  # noqa: N802
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> str | None:
        """Sets data for the given role and section in the header with the specified orientation.

        Args:
            section (int): For horizontal headers, the section number corresponds to the column
                number. For vertical headers, the section number corresponds to the row number.
            orientation (Qt.Orientation): The header orientation.
            role (int, optional): Data rendered as text. Defaults to Qt.ItemDataRole.DisplayRole.

        Returns:
            str | None: Header data.
        """
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.df.columns[section])
        if orientation == Qt.Orientation.Vertical:
            return str(section)
        return None


def create_dataframe_tab(df: pl.DataFrame, _: str) -> QWidget:
    """Create a QWidget containing a QTableView to display the DataFrame.

    Args:
        df (pl.DataFrame): Data to be displayed in the tab.
        _ (str): The tab label.

    Returns:
        QWidget: A QWidget containing a QTableView to display the DataFrame.
    """
    widget: QWidget = QWidget()
    layout: QVBoxLayout = QVBoxLayout(widget)
    table_view: QTableView = QTableView()
    model: MyTable = MyTable(df)
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
        """Initialize class variables."""
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

        self.setWindowTitle("pyGEONOSIS")
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

        top_panel: QWidget = self.create_top_panel()
        main_layout.addWidget(top_panel)

        main_panel: QWidget = self.create_main_panel()
        main_layout.addWidget(main_panel, stretch=1)

        bottom_panel: QWidget = self.create_bottom_panel()
        main_layout.addWidget(bottom_panel)

    def create_top_panel(self) -> QWidget:  # noqa: PLR0915
        """Create the top panel with band ranges, broadening, granularity, and run controls.

        Returns:
            QWidget: The top panel widget.
        """
        top_widget: QWidget = QWidget()
        layout: QHBoxLayout = QHBoxLayout(top_widget)

        # Specific bands or band range inputs.
        group_bands: QGroupBox = QGroupBox("Bands")
        bands_layout: QVBoxLayout = QVBoxLayout(group_bands)

        # TODO: 25/04/09 - Changing the mode from specific bands to band ranges resizes the UI,
        #       which is really annoying. Fix this in the future.

        band_selection_layout: QHBoxLayout = QHBoxLayout()
        self.radio_specific_bands: QRadioButton = QRadioButton("Specific Bands")
        self.radio_specific_bands.setChecked(True)
        self.radio_band_ranges: QRadioButton = QRadioButton("Band Ranges")
        band_selection_layout.addWidget(self.radio_specific_bands)
        band_selection_layout.addWidget(self.radio_band_ranges)
        bands_layout.addLayout(band_selection_layout)

        self.radio_specific_bands.toggled.connect(self.toggle_band_input_method)

        # Specific band input.
        self.specific_bands_container: QWidget = QWidget()
        specific_bands_layout: QVBoxLayout = QVBoxLayout(self.specific_bands_container)
        specific_bands_layout.setContentsMargins(0, 0, 0, 0)

        band_layout: QHBoxLayout = QHBoxLayout()
        band_label: QLabel = QLabel("Bands:")
        self.band_line_edit: QLineEdit = QLineEdit(DEFAULT_BANDS)
        band_layout.addWidget(band_label)
        band_layout.addWidget(self.band_line_edit)
        specific_bands_layout.addLayout(band_layout)

        # Band range input.
        self.band_ranges_container: QWidget = QWidget()
        band_ranges_layout: QGridLayout = QGridLayout(self.band_ranges_container)
        band_ranges_layout.setContentsMargins(0, 0, 0, 0)

        v_up_min_label: QLabel = QLabel("v' min:")
        self.v_up_min_spinbox: QSpinBox = QSpinBox()
        self.v_up_min_spinbox.setRange(0, 99)
        self.v_up_min_spinbox.setValue(0)

        v_up_max_label: QLabel = QLabel("v' max:")
        self.v_up_max_spinbox: QSpinBox = QSpinBox()
        self.v_up_max_spinbox.setRange(0, 99)
        self.v_up_max_spinbox.setValue(10)

        v_lo_min_label: QLabel = QLabel("v'' min:")
        self.v_lo_min_spinbox: QSpinBox = QSpinBox()
        self.v_lo_min_spinbox.setRange(0, 99)
        self.v_lo_min_spinbox.setValue(5)

        v_lo_max_label: QLabel = QLabel("v'' max:")
        self.v_lo_max_spinbox: QSpinBox = QSpinBox()
        self.v_lo_max_spinbox.setRange(0, 99)
        self.v_lo_max_spinbox.setValue(5)

        band_ranges_layout.addWidget(v_up_min_label, 0, 0)
        band_ranges_layout.addWidget(self.v_up_min_spinbox, 0, 1)
        band_ranges_layout.addWidget(v_up_max_label, 0, 2)
        band_ranges_layout.addWidget(self.v_up_max_spinbox, 0, 3)

        band_ranges_layout.addWidget(v_lo_min_label, 1, 0)
        band_ranges_layout.addWidget(self.v_lo_min_spinbox, 1, 1)
        band_ranges_layout.addWidget(v_lo_max_label, 1, 2)
        band_ranges_layout.addWidget(self.v_lo_max_spinbox, 1, 3)

        self.band_ranges_container.hide()

        bands_layout.addWidget(self.specific_bands_container)
        bands_layout.addWidget(self.band_ranges_container)

        layout.addWidget(group_bands)

        # Broadening inputs.
        group_broadening: QGroupBox = QGroupBox("Instrument Broadening [nm]")
        broadening_layout: QHBoxLayout = QHBoxLayout(group_broadening)
        self.inst_broadening_spinbox: MyDoubleSpinBox = MyDoubleSpinBox()
        self.inst_broadening_spinbox.setValue(DEFAULT_BROADENING)
        broadening_layout.addWidget(self.inst_broadening_spinbox)

        checkbox_layout: QHBoxLayout = QHBoxLayout()
        self.checkbox_instrument: QCheckBox = QCheckBox("Instrument FWHM")
        self.checkbox_instrument.setChecked(True)
        self.checkbox_doppler: QCheckBox = QCheckBox("Doppler")
        self.checkbox_doppler.setChecked(True)
        self.checkbox_natural: QCheckBox = QCheckBox("Natural")
        self.checkbox_natural.setChecked(True)
        self.checkbox_collisional: QCheckBox = QCheckBox("Collisional")
        self.checkbox_collisional.setChecked(True)
        self.checkbox_predissociation: QCheckBox = QCheckBox("Predissociation")
        self.checkbox_predissociation.setChecked(True)
        checkbox_layout.addWidget(self.checkbox_instrument)
        checkbox_layout.addWidget(self.checkbox_doppler)
        checkbox_layout.addWidget(self.checkbox_natural)
        checkbox_layout.addWidget(self.checkbox_collisional)
        checkbox_layout.addWidget(self.checkbox_predissociation)
        broadening_layout.addLayout(checkbox_layout)
        layout.addWidget(group_broadening)

        # Granularity input.
        group_granularity: QGroupBox = QGroupBox("Granularity")
        granularity_layout: QHBoxLayout = QHBoxLayout(group_granularity)
        self.granularity_spinbox: QSpinBox = QSpinBox()
        self.granularity_spinbox.setMaximum(10000000)
        self.granularity_spinbox.setValue(DEFAULT_GRANULARITY)
        granularity_layout.addWidget(self.granularity_spinbox)
        layout.addWidget(group_granularity)

        # Rotational line input.
        group_lines: QGroupBox = QGroupBox("Rotational Lines")
        lines_layout: QHBoxLayout = QHBoxLayout(group_lines)
        self.num_lines_spinbox: QSpinBox = QSpinBox()
        self.num_lines_spinbox.setMaximum(10000)
        self.num_lines_spinbox.setValue(DEFAULT_LINES)
        lines_layout.addWidget(self.num_lines_spinbox)
        layout.addWidget(group_lines)

        # Run controls.
        group_run: QGroupBox = QGroupBox("Actions")
        run_layout: QHBoxLayout = QHBoxLayout(group_run)
        self.run_button: QPushButton = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.add_simulation)
        run_layout.addWidget(self.run_button)
        self.open_button: QPushButton = QPushButton("Open File")
        self.open_button.clicked.connect(self.add_sample)
        run_layout.addWidget(self.open_button)
        self.export_button: QPushButton = QPushButton("Export Table")
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
        main_widget: QWidget = QWidget()
        layout: QHBoxLayout = QHBoxLayout(main_widget)

        # Tabs containing tables.
        self.tab_widget: QTabWidget = QTabWidget()
        empty_df: pl.DataFrame = pl.DataFrame()
        empty_tab: QWidget = create_dataframe_tab(empty_df, "v'-v''")
        self.tab_widget.addTab(empty_tab, "v'-v''")
        layout.addWidget(self.tab_widget, stretch=1)

        # Plot area.
        self.plot_widget: pg.PlotWidget = pg.PlotWidget()

        self.plot_widget.addLegend(offset=(0, 1))
        self.plot_widget.setAxisItems({"top": WavenumberAxis(orientation="top")})
        self.plot_widget.setLabel("top", "Wavenumber, ν [cm<sup>-1</sup>]")
        self.plot_widget.setLabel("bottom", "Wavelength, λ [nm]")
        self.plot_widget.setLabel("left", "Intensity, I [a.u.]")
        self.plot_widget.setLabel("right", "Intensity, I [a.u.]")

        self.plot_widget.setXRange(100, 200)

        layout.addWidget(self.plot_widget, stretch=2)
        self.legend = self.plot_widget.addLegend()

        return main_widget

    def create_bottom_panel(self) -> QWidget:  # noqa: PLR0915
        """Create the bottom panel with temperature, pressure, and combo selections.

        Returns:
            QWidget: The bottom panel widget.
        """
        bottom_widget: QWidget = QWidget()
        layout: QHBoxLayout = QHBoxLayout(bottom_widget)

        # Temperature and pressure inputs.
        entries_widget: QWidget = QWidget()
        entries_layout: QGridLayout = QGridLayout(entries_widget)

        # Equilibrium temperature input.
        self.temp_label: QLabel = QLabel("Temperature [K]:")
        self.temp_spinbox: MyDoubleSpinBox = MyDoubleSpinBox()
        self.temp_spinbox.setValue(DEFAULT_TEMPERATURE)
        entries_layout.addWidget(self.temp_label, 0, 0)
        entries_layout.addWidget(self.temp_spinbox, 0, 1)

        # Nonequilibrium temperature input.
        self.temp_trn_label: QLabel = QLabel("Translational Temp. [K]:")
        self.temp_trn_spinbox: MyDoubleSpinBox = MyDoubleSpinBox()
        self.temp_trn_spinbox.setValue(DEFAULT_TEMPERATURE)
        self.temp_elc_label: QLabel = QLabel("Electronic Temp. [K]:")
        self.temp_elc_spinbox: MyDoubleSpinBox = MyDoubleSpinBox()
        self.temp_elc_spinbox.setValue(DEFAULT_TEMPERATURE)
        self.temp_vib_label: QLabel = QLabel("Vibrational Temp. [K]:")
        self.temp_vib_spinbox: MyDoubleSpinBox = MyDoubleSpinBox()
        self.temp_vib_spinbox.setValue(DEFAULT_TEMPERATURE)
        self.temp_rot_label: QLabel = QLabel("Rotational Temp. [K]:")
        self.temp_rot_spinbox: MyDoubleSpinBox = MyDoubleSpinBox()
        self.temp_rot_spinbox.setValue(DEFAULT_TEMPERATURE)

        # Show equilibrium temperature input by default.
        for widget in (
            self.temp_trn_label,
            self.temp_trn_spinbox,
            self.temp_elc_label,
            self.temp_elc_spinbox,
            self.temp_vib_label,
            self.temp_vib_spinbox,
            self.temp_rot_label,
            self.temp_rot_spinbox,
        ):
            widget.hide()

        entries_layout.addWidget(self.temp_trn_label, 0, 2)
        entries_layout.addWidget(self.temp_trn_spinbox, 0, 3)
        entries_layout.addWidget(self.temp_elc_label, 0, 4)
        entries_layout.addWidget(self.temp_elc_spinbox, 0, 5)
        entries_layout.addWidget(self.temp_vib_label, 0, 6)
        entries_layout.addWidget(self.temp_vib_spinbox, 0, 7)
        entries_layout.addWidget(self.temp_rot_label, 0, 8)
        entries_layout.addWidget(self.temp_rot_spinbox, 0, 9)

        # Pressure input.
        pressure_label = QLabel("Pressure [Pa]:")
        self.pressure_spinbox = MyDoubleSpinBox()
        self.pressure_spinbox.setValue(DEFAULT_PRESSURE)
        entries_layout.addWidget(pressure_label, 1, 0)
        entries_layout.addWidget(self.pressure_spinbox, 1, 1)

        layout.addWidget(entries_widget)

        # Simulation and plot parameters.
        combos_widget: QWidget = QWidget()
        combos_layout: QGridLayout = QGridLayout(combos_widget)

        # Temperature Type.
        temp_type_label: QLabel = QLabel("Temperature Type:")
        self.temp_type_combo: QComboBox = QComboBox()
        self.temp_type_combo.addItems(["Equilibrium", "Nonequilibrium"])
        self.temp_type_combo.currentTextChanged.connect(self.switch_temp_mode)
        combos_layout.addWidget(temp_type_label, 0, 0)
        combos_layout.addWidget(self.temp_type_combo, 0, 1)

        # Simulation Type.
        sim_type_label: QLabel = QLabel("Simulation Type:")
        self.sim_type_combo: QComboBox = QComboBox()
        self.sim_type_combo.addItems(["Absorption", "Emission"])
        combos_layout.addWidget(sim_type_label, 1, 0)
        combos_layout.addWidget(self.sim_type_combo, 1, 1)

        # Plot Type.
        plot_type_label: QLabel = QLabel("Plot Type:")
        self.plot_type_combo: QComboBox = QComboBox()
        self.plot_type_combo.addItems(["Line", "Line Info", "Convolve Separate", "Convolve All"])
        combos_layout.addWidget(plot_type_label, 2, 0)
        combos_layout.addWidget(self.plot_type_combo, 2, 1)

        layout.addWidget(combos_widget)

        return bottom_widget

    def add_sample(self) -> None:
        """Open a CSV file and add a new tab showing the contents."""
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
        """Parse comma-separated band ranges from user input.

        Returns:
            list[tuple[int, int]]: A list of vibrational bands, e.g. [(1, 2), (3, 4)].
        """
        band_ranges_str: str = self.band_line_edit.text()
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
        """Run a simulation instance, then update the plot and table tabs."""
        start_time: float = time.time()

        # TODO: 25/04/14 - Split this method up. Process the temperature mode, parse the bands,
        #       create the simulation, plot the data, and finally make a new tab in the table.

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

        if self.radio_specific_bands.isChecked():
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
            v_up_min: int = self.v_up_min_spinbox.value()
            v_up_max: int = self.v_up_max_spinbox.value()
            v_lo_min: int = self.v_lo_min_spinbox.value()
            v_lo_max: int = self.v_lo_max_spinbox.value()

            if v_up_min > v_up_max or v_lo_min > v_lo_max:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Invalid band range: min value cannot be greater than max value.",
                    QMessageBox.StandardButton.Ok,
                )
                return

            # Generate band combinations based on the given ranges.
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

        rot_lvls: NDArray[np.int64] = np.arange(0, self.num_lines_spinbox.value())

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
        if not hasattr(model, "df"):
            QMessageBox.information(
                self,
                "Error",
                "This table does not support CSV export.",
                QMessageBox.StandardButton.Ok,
            )
            return

        df: pl.DataFrame = model.df

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

    def __init__(self, *args, **kwargs) -> None:
        """Initialize class variables."""
        super().__init__(*args, **kwargs)

    def tickStrings(self, wavelengths: list[float], *_) -> list[str]:  # noqa: N802
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


class LoadingWorker(QObject):
    """Worker class to handle initialization tasks and report progress."""

    progress_updated: Signal = Signal(int, str)
    finished: Signal = Signal()

    def __init__(self) -> None:
        """Initialize class variables."""
        super().__init__()

    def run_initialization(self) -> None:
        """Run all initialization tasks sequentially.

        The emitters are completely arbitrary and don't represent the actual state of the program,
        they're just here to make loading look a bit cooler.
        """
        self.progress_updated.emit(0, "Starting application...")

        self.progress_updated.emit(20, "Initializing UI components...")

        self.progress_updated.emit(60, "Preparing main window...")
        self.main_window: GUI = GUI()

        self.progress_updated.emit(90, "Almost ready...")

        self.progress_updated.emit(100, "Done!")
        self.finished.emit()


class SplashScreen(QWidget):
    """A custom splash screen with progress bar for application startup."""

    def __init__(self) -> None:
        """Initialize class variables."""
        super().__init__()

        # Remove window frame and make the background transparent.
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(600, 400)
        self.center()

        self.progress_value: int = 0
        self.status_message: str = "Loading..."

        self.worker: LoadingWorker = LoadingWorker()
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.finished.connect(self.on_initialization_complete)

        # Start initialization process with a slight delay to allow the splash screen to show.
        QTimer.singleShot(1000, self.worker.run_initialization)

    def center(self) -> None:
        """Center the window on the screen."""
        qr: QRect = self.frameGeometry()
        qp: QPoint = self.screen().availableGeometry().center()
        qr.moveCenter(qp)
        self.move(qr.topLeft())

    def update_progress(self, value: int, message: str) -> None:
        """Update the progress bar value and message."""
        self.progress_value = value
        self.status_message = message
        self.repaint()

    def on_initialization_complete(self) -> None:
        """Handle completion of initialization."""
        self.main_window: GUI = self.worker.main_window
        self.close()
        self.main_window.show()

    def paintEvent(self, _: QPaintEvent) -> None:  # noqa: N802
        """Custom paint event to draw the splash screen.

        Args:
            _ (QPaintEvent): Contains event parameters for paint events.
        """
        painter: QPainter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Rounded rectangular background.
        painter.setBrush(QColor(30, 30, 30))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 20, 20)

        # Application title.
        painter.setPen(QColor(255, 255, 255))
        title_font: QFont = QFont("Arial", 24, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(
            0, 60, self.width(), 40, Qt.AlignmentFlag.AlignCenter, "pyGEONOSIS"
        )

        # Subtitle.
        subtitle_font: QFont = QFont("Arial", 12)
        painter.setFont(subtitle_font)
        painter.drawText(
            0,
            100,
            self.width(),
            30,
            Qt.AlignmentFlag.AlignCenter,
            "Python GEnerated Oxygen and Nitric Oxide SImulated Spectra",
        )

        # Status text.
        status_font: QFont = QFont("Arial", 10)
        painter.setFont(status_font)
        painter.drawText(
            20,
            self.height() - 50,
            self.width() - 40,
            20,
            Qt.AlignmentFlag.AlignCenter,
            self.status_message,
        )

        # Progress bar background.
        painter.setBrush(QColor(50, 50, 50))
        painter.drawRoundedRect(20, self.height() - 30, self.width() - 40, 10, 5, 5)

        # Progress bar.
        progress_width: int = int((self.width() - 40) * (self.progress_value / 100))
        painter.setBrush(QColor(0, 123, 255))
        painter.drawRoundedRect(20, self.height() - 30, progress_width, 10, 5, 5)

        # O2 molecule visualization.
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(QColor(200, 200, 255, 150))

        # Two oxygen atoms.
        center_x: int = self.width() // 2
        center_y: int = self.height() // 2
        radius: int = 40
        distance: int = 100

        # Left oxygen atom.
        painter.drawEllipse(
            center_x - distance // 2 - radius // 2, center_y - radius // 2, radius, radius
        )

        # Right oxygen atom.
        painter.drawEllipse(
            center_x + distance // 2 - radius // 2, center_y - radius // 2, radius, radius
        )

        # Bond between atoms.
        painter.setPen(QPen(QColor(150, 150, 255), 4))
        painter.drawLine(
            center_x - distance // 2 + radius // 2,
            center_y,
            center_x + distance // 2 - radius // 2,
            center_y,
        )


def main() -> None:
    """Entry point."""
    app: QApplication = QApplication(sys.argv)

    app_icon: QIcon = QIcon(str(utils.get_data_path("img", "icon.ico")))
    app.setWindowIcon(app_icon)

    splash: SplashScreen = SplashScreen()
    splash.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
