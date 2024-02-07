"""
author: Jiatu Liu

issues:
    C:\Programfiles\Anaconda3-2020-07\envs\integration\lib\site-packages\pyqtgraph\
        GraphicsScene\GraphicsScene.py:278: RuntimeWarning: Error sending hover event:
    
    WARNING:silx.opencl.common:Unable to import pyOpenCl. Please install it from: 
        https://pypi.org/project/pyopencl
notes:
    doesn't apply to linux due to the difference between \\ and /
        
"""

import os
import logging # this is interesting
logging.getLogger('fabio').setLevel(logging.ERROR)
logging.getLogger('pyFAI').setLevel(logging.ERROR)
import re
import glob
import sys
import traceback
import time
import numpy as np
np.seterr(divide = 'ignore') 
# import h5py
import fabio
# import tifffile # can also handle tags
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import multiprocessing
import pyFAI
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph.dockarea
import pyqtgraph as pg
from draggabletabwidget_new import *
import faulthandler
import ctypes
import platform
def make_dpi_aware():
    if int(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    # result = pyqtSignal(object)
    # progress = pyqtSignal(int)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress


    @pyqtSlot()
    def run(self):
        self.fn(*self.args, **self.kwargs)
        # Retrieve args/kwargs here; and fire processing using them
        try:
            self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()  # Done



class Dataclass():
    def __init__(self):
        self.data = None
        self.pen = 'r'
        self.symbol = None
        self.symbolsize = None
        self.symbolbrush = 'b'
        self.image = None

class Paraclass():
    def __init__(self, **kwargs):
        for key in kwargs:
            self.identity = key # only accept one key now!
            if key == 'values':
                self.setvalue = kwargs[key][0]
                self.upper = kwargs[key][2]
                self.lower = kwargs[key][1]
                self.step = kwargs[key][3]
                # bear in mind that the actual step is 1, the nominal value needs a multiplier, 10

            if key == 'strings':
                self.choice = kwargs[key][0]
                self.choices = kwargs[key][1]

class DockGraph(): 
    # to do: it is maybe appropriate to put this in ShowData, 
    # on a second thought, no! e.g. slider should go under each methodobj! so maybe in Method_Base
    # on a third thought, no! a methodobj can have more than one DockGraph, maybe with more effort...
    # top level docking graph widget
    def __init__(self, name):
        self.label = name # e.g. xas, xrd,...
        self.tabdict = {} # a dic for this level tab objects, important!

    def gendock(self, winobj, methodobj):        
        # self.slider = QSlider(Qt.Horizontal) # put into Method_Base
        methodobj.slider.setObjectName(self.label)
        methodobj.slider.valueChanged.connect(winobj.update_timepoints)
        # to do: put update_timepoints into individual Method, no! stay in ShowData

        horilayout = QHBoxLayout()
        # horilayout.setMaximumHeight(int(winobj.screen_height * .1)) # QHBoxLayout has no setM_H_
        horilayout.addWidget(methodobj.slider)
        # self.sliderlabel = QLabel() # put into Method_Base
        
        horilayout.addWidget(methodobj.sliderlabel)
        
        methodobj.mem.setObjectName(self.label)
        methodobj.mem.clicked.connect(winobj.memorize_curve)
        horilayout.addWidget(methodobj.mem)
        methodobj.clr.setObjectName(self.label)
        methodobj.clr.clicked.connect(winobj.clear_curve)
        horilayout.addWidget(methodobj.clr)
        # to do: another pair of mem and clr for global compar.
        
        self.dockobj = QDockWidget(self.label, winobj)
        self.dockobj.setMinimumWidth(int(winobj.screen_width * .3))
        self.dockobj.setMinimumHeight(int(winobj.screen_height * .5))
        winobj.addDockWidget(Qt.RightDockWidgetArea, self.dockobj)
        if len(winobj.gdockdict) > 3: # only accommodate two docks
            self.dockobj.setFloating(True)
        else: 
            self.dockobj.setFloating(False)

        self.docktab = DraggableTabWidget()
        vertlayout = QVBoxLayout()
        vertlayout.addLayout(horilayout)
        vertlayout.addWidget(self.docktab)
        vertwidget = QWidget()
        vertwidget.setLayout(vertlayout)
        self.dockobj.setWidget(vertwidget)

    def deldock(self, winobj):
        winobj.removeDockWidget(self.dockobj)
        DraggableTabBar.dragging_widget_ = None
        DraggableTabBar.tab_bar_instances_ = [] # these two lines work?!
        del self.docktab # these two lines caused fatal error!!! access violation ?!
        del self.dockobj # be careful when using del ?!

    def gencontroltab(self, winobj):
        self.tooltab = QToolBox()
        # self.tooltab.setObjectName(self.label)
        winobj.controltabs.addTab(self.tooltab,self.label)

    def delcontroltab(self, winobj):
        index = winobj.controltabs.indexOf(self.tooltab)
        winobj.controltabs.removeTab(index)
        DraggableTabBar.dragging_widget_ = None
        DraggableTabBar.tab_bar_instances_ = []  # these two lines work!!!
        # del self.tooltab



class TabGraph():
    def __init__(self, name):
        self.label = name # e.g. raw, norm,...

    def mouseMoved(self, evt): # surprise!
        self.mousePoint = self.tabplot.vb.mapSceneToView(evt) # not evt[0]
        has_image = False
        if len(self.tabplot.items) > 0: # read z value of an image
            for item in self.tabplot.items:
                if hasattr(item, 'image'):
                    has_image = True
                    i_x = int(self.mousePoint.x())
                    i_y = int(self.mousePoint.y())
                    if (0 <= i_x < item.image.shape[0]) and (0 <= i_y < item.image.shape[1]):
                        # self.tabplot_label_z.setText("<span style='font-size: 10pt; color: black'> z = %0.3f</span>" %
                        #                              item.image[i_x, i_y])
                        self.tabplot_label_z.setText("<span> log10(z) = {:10.3f} </span>".format(item.image[i_x, i_y]))
                        self.tabplot_label.setText("<span> x = {:10.3f}, y = {:10.3f} </span>".format
                                                   (self.mousePoint.x(), self.mousePoint.y()))

            if not has_image:
                self.tabplot_label.setText("<span> x = %0.3f, y = %0.3f </span>" % (self.mousePoint.x(), self.mousePoint.y()))

    def gentab(self, dockobj, methodobj): # generate a tab for a docking graph
        if self.label == 'time series' and dockobj.label[0:3] == 'xrd':
            self.graphtab = MyGraphicsLayoutWidget(self, methodobj)
        else:
            self.graphtab = pg.GraphicsLayoutWidget()

        dockobj.docktab.addTab(self.graphtab, self.label)
        self.tabplot_label = pg.LabelItem(justify='left')
        # self.tabplot_label.setFixedWidth(100)
        self.tabplot_label_z = pg.LabelItem(justify='left')
        self.graphtab.addItem(self.tabplot_label_z, row=0, col=0)
        self.graphtab.addItem(self.tabplot_label, row=1, col=0)
        self.tabplot = self.graphtab.addPlot(row=2, col=0)
        # pg.SignalProxy(self.tabplot.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved) # this is outdated!
        self.tabplot.scene().sigMouseMoved.connect(self.mouseMoved) # this is correct !
        self.tabplot.setLabel('bottom',methodobj.axislabel[self.label]['bottom'])
        self.tabplot.setLabel('left', methodobj.axislabel[self.label]['left'])
        if methodobj.axislabel[self.label]['left'] not in ['Data number', '<-- Data number --']:
            self.tabplot.addLegend(labelTextSize='9pt')

    def deltab(self, dockobj):
        # print('del graph tab 1')
        index = dockobj.docktab.indexOf(self.graphtab)
        dockobj.docktab.removeTab(index)
        # print('del graph tab 2')
        # del self.graphtab
        # print('del graph tab 3')

    def gencontrolitem(self, dockobj):
        self.itemwidget = QWidget()
        self.itemwidget.setObjectName(self.label)
        self.itemwidget.setAccessibleName(dockobj.label)
        self.itemlayout = QVBoxLayout() # add control options to this layout
        self.itemwidget.setLayout(self.itemlayout)
        dockobj.tooltab.addItem(self.itemwidget, self.label)

    def delcontrolitem(self, dockobj):
        self.itemlayout.setParent(None)
        # index = dockobj.tooltab.indexOf(self.itemwidget)
        # dockobj.tooltab.removeItem(index)
        self.itemwidget.setParent(None)
        # del self.itemlayout
        # del self.itemwidget

    def delcurvechecks(self, tabname, methodobj): # curvedict is a dict for all curve checkboxes
        # actions
        if tabname in methodobj.actions:
            for key in methodobj.actions[tabname]:
                methodobj.actwidgets[tabname][key].setParent(None)

        # parameters
        if tabname in methodobj.parameters: # normalized, chi(k)
            for key in methodobj.parameters[tabname]: # rbkg, kmin,...
                methodobj.parawidgets[tabname][key].setParent(None)
                methodobj.paralabel[tabname][key].setParent(None)

        # checkboxes
        for key in methodobj.availablecurves[tabname]:
            if methodobj.curvedict[tabname][key].isChecked():
                methodobj.curvedict[tabname][key].setChecked(False)

            methodobj.curvedict[tabname][key].setParent(None)

        # LineEdit
        if tabname in methodobj.linedit:
            for key in methodobj.linedit[tabname]:
                # methodobj.linewidgets[tabname][key].setParent(None)
                label = self.range_select.labelForField(methodobj.linewidgets[tabname][key])
                if label != None: label.setParent(None)
                methodobj.linewidgets[tabname][key].setParent(None)

            self.itemlayout.removeItem(self.itemlayout.itemAt(1))
        # the spacer:
        self.itemlayout.removeItem(self.itemlayout.itemAt(0))

    # tabname, e.g. raw, norm,...; methodobj, e.g. an XAS obj; for 'I0', 'I1',...
    def curvechecks(self, tabname, methodobj, winobj):
        # checkboxes
        for key in methodobj.availablecurves[tabname]:
            methodobj.curvedict[tabname][key] = QCheckBox(key)
            methodobj.curvedict[tabname][key].stateChanged.connect(winobj.graphcurve)
            self.itemlayout.addWidget(methodobj.curvedict[tabname][key])

        # refinement case
        if tabname == 'refinement single':
            for key in methodobj.availablecurves[tabname]:
                if key[0:5] == 'phase': methodobj.curvedict[tabname][key].setChecked(True)

        # parameters
        if tabname in methodobj.parameters:
            methodobj.parawidgets[tabname] = {}
            methodobj.paralabel[tabname] = {}
            for key in methodobj.parameters[tabname]: # rbkg, kmin,...
                temppara = methodobj.parameters[tabname][key]
                if temppara.identity == 'values':
                    methodobj.parawidgets[tabname][key] = QSlider(Qt.Horizontal)
                    tempwidget = methodobj.parawidgets[tabname][key]
                    tempwidget.setObjectName(key)
                    tempwidget.setRange(0, int((temppara.upper - temppara.lower) / temppara.step))
                    # tempwidget.setSingleStep(temppara.step)
                    tempwidget.setValue(int((temppara.setvalue - temppara.lower) / temppara.step))
                    tempwidget.valueChanged.connect(winobj.update_parameters)
                    methodobj.paralabel[tabname][key] = QLabel(key + ':' + '{:.1f}'.format(temppara.setvalue))
                else:
                    methodobj.parawidgets[tabname][key] = QComboBox()
                    tempwidget = methodobj.parawidgets[tabname][key]
                    tempwidget.setObjectName(key)
                    tempwidget.addItems(temppara.choices)
                    tempwidget.currentTextChanged.connect(winobj.update_parameters)
                    methodobj.paralabel[tabname][key] = QLabel(key)

                self.itemlayout.addWidget(methodobj.paralabel[tabname][key])
                self.itemlayout.addWidget(tempwidget)

        # actions
        if tabname in methodobj.actions:
            methodobj.actwidgets[tabname] = {}
            for key in methodobj.actions[tabname]:
                methodobj.actwidgets[tabname][key] = QPushButton(key)
                if key[-1] == ')':
                    methodobj.actwidgets[tabname][key].setShortcut(key[-7:-1])
                # tempfunc = getattr(methodobj, methodobj.actions[tabname][key])

                if key == 'update y,z range (Ctrl+0)': # for update the range of 2D plot
                    methodobj.actwidgets[tabname][key].clicked.connect(
                        lambda state, t = tabname, k = key: methodobj.actions[t][k](winobj, methodobj, tabname))  # e,g, MainWin, XAS, munorm-T
                    # what is this state!!!???
                    # The QPushButton.clicked signal emits an argument that indicates the state of the button.
                    # When you connect to your lambda slot, the optional argument you assign to is being overwritten by the state of the button.
                    # This way the button state is ignored and the correct value is passed to your method.

                else:
                    methodobj.actwidgets[tabname][key].clicked.connect(
                        lambda state, t = tabname, k = key: methodobj.actions[t][k](winobj))  # shock !!!
                        # lambda state, k = key: methodobj.actions[tabname][k](winobj)) # shock !!!
                self.itemlayout.addWidget(methodobj.actwidgets[tabname][key])

        # lineEdit
        if tabname in methodobj.linedit:
            methodobj.linewidgets[tabname] = {}
            self.range_select = QFormLayout()
            for key in methodobj.linedit[tabname]:
                methodobj.linewidgets[tabname][key] = QLineEdit(methodobj.linedit[tabname][key])
                self.range_select.addRow(key, methodobj.linewidgets[tabname][key])
                
            self.itemlayout.addLayout(self.range_select)

        self.curvespacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.itemlayout.addItem(self.curvespacer)


class Methods_Base():
    def __init__(self):
        # self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        self.slider = QSlider(Qt.Horizontal) # each methodobj has a slider and sliderlabel
        self.sliderlabel = QLabel()
        self.slider_v = 0
        self.mem = QPushButton("Memorize")
        # methodobj.mem.setShortcut('Ctrl+M')
        self.clr = QPushButton("Clear")
        # methodobj.clr.setShortcut('Ctrl+C')
        self.availablemethods = []
        self.availablecurves = {}
        self.rawfilename = []
        self.axislabel = {}
        self.dynamictitle = []
        self.checksdict = {}  # QCheckBox obj for raw, norm,...
        self.curvedict = {}  # QCheckBox obj for curves belong to raw, norm,...
        # the first element in the time series list, the following elements can be memorized
        self.data_timelist = []
        self.data_timelist.append({})
        # self.data_timelist = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
        self.curve_timelist = []
        self.curve_timelist.append({})  # the curve, corresponding to each data obj
        # self.curve_timelist = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}] # the curves
        self.colorhue = {}
        self.index = 0 # based on this to update the curves/data or not
        self.index_ref = -1 # based on this to update the curves/data or not
        self.update = True
        # self.timediff = []  # based on this to decide the index from the slidervalue
        self.parawidgets = {} # to control parameters after curve checkboxes
        self.actwidgets = {}
        self.linewidgets = {}
        self.parameters = {} # control parameters
        self.actions = {}
        self.linedit = {}
        self.paralabel = {} # control para labels widget
        self.process_started = False
        self.maxhue = 50  # 100 colorhues
        self.huestep = 2 # color rotation increment
        # self.data_num = 0 # to do: maybe a list collect all data?
        self.int_data = {}
        self.raw_data = {}
        # self.ts_list = [] # may put somewhere else
        # self.ts_text = [] # this has caused the fatal error: Windows access violation
    

    def plot_pointer(self, tabname, p_x, p_y, symbol, size):
        self.data_timelist[0][tabname]['pointer'].data = np.array([[p_x], [p_y]]).transpose()
        self.data_timelist[0][tabname]['pointer'].symbol = symbol
        self.data_timelist[0][tabname]['pointer'].symbolsize = size

    def ini_data_curve_color(self):
        for key in self.availablemethods: # raw, norm,...
            self.colorhue[key] = len(self.availablecurves[key]) * self.huestep # ini the colorhue for time series
            self.curvedict[key] = {}  # QCheckBox obj for curves belong to raw, norm,...
            self.data_timelist[0][key] = {}  # e.g. self.data_timelist[0]['raw']
            self.curve_timelist[0][key] = {}  # just initialize here
            for subkey in self.availablecurves[key]:
                self.data_timelist[0][key][subkey] = Dataclass()  # e.g. self.data_timelist[0]['raw']['I0']


    def data_update(self, slidervalue): # to do: may simplify to data_process?
        # colors and processes
        self.slider_v = slidervalue
        for key in self.data_timelist[0]:
            nstep = 0
            for subkey in self.data_timelist[0][key]:
                if self.data_timelist[0][key][subkey].pen is not None:
                    pen = pg.mkPen(pg.intColor(nstep * self.huestep, self.maxhue), width=1.5)
                    self.data_timelist[0][key][subkey].pen = pen
                    nstep += 1

            self.data_process(False)

    
    def data_curve_copy(self, data): # data and curve are, e.g., the first obj in the data_timelist and curve_timelist
        # used for copying ...list[0] to next list element.
        newdata = {}
        newcurve = {}
        for key in self.availablemethods: # raw, norm,...
            newdata[key] = {}
            newcurve[key] = {}
            for subkey in self.availablecurves[key]:
                newdata[key][subkey] = Dataclass()
                # this error proof should always be present to separate, say curve from image
                if subkey in data[key]:
                    if hasattr(data[key][subkey], 'data'):
                        if data[key][subkey].data is not None:
                            newdata[key][subkey].data = data[key][subkey].data
                            if data[key][subkey].symbol is not None:
                                newdata[key][subkey].symbol = data[key][subkey].symbol
                                newdata[key][subkey].symbolsize = data[key][subkey].symbolsize
                                newdata[key][subkey].symbolbrush = pg.intColor(self.colorhue[key], self.maxhue)
                            elif data[key][subkey].pen is not None:
                                newdata[key][subkey].pen = pg.mkPen(pg.intColor(self.colorhue[key], self.maxhue), width=1.5)

                            self.colorhue[key] += self.huestep

        return newdata, newcurve

    def tabchecks(self, winobj, method): # generate all checks for the second level methods: raw, norm,.
        for key in self.availablemethods: # raw, norm,...
            self.checksdict[key] = QCheckBox(key) # e.g. self.checksdict['raw']
            self.checksdict[key].stateChanged.connect(winobj.graph_tab)
            winobj.subcboxverti[method].addWidget(self.checksdict[key])

        self.btn_dynamic = QPushButton('int. w. watchdog (Ctrl+1)')
        self.btn_dynamic.setShortcut('Ctrl+1')
        self.btn_dynamic.clicked.connect(lambda: self.int_dynamic(winobj))
        self.btn_static = QPushButton('int. w/o watchdog (Ctrl+2)')
        self.btn_static.setShortcut('Ctrl+2')
        self.btn_static.clicked.connect(lambda: self.int_static(winobj))
        winobj.subcboxverti[method].addWidget(self.btn_dynamic)
        winobj.subcboxverti[method].addWidget(self.btn_static)
        self.itemspacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        winobj.subcboxverti[method].addItem(self.itemspacer)

    def deltabchecks(self, winobj, method): # del above checks
        for key in self.availablemethods:  # raw, norm,...
            if self.checksdict[key].isChecked(): self.checksdict[key].setChecked(False)
            winobj.subcboxverti[method].removeWidget(self.checksdict[key])

        self.btn_dynamic.setParent(None)
        self.btn_static.setParent(None)
        winobj.subcboxverti[method].removeWidget(self.btn_dynamic)
        winobj.subcboxverti[method].removeWidget(self.btn_static)
        winobj.subcboxverti[method].removeItem(self.itemspacer)
        # winobj.delslider()  # del the slider # to do: may be need to del the slider?
        self.checksdict = {}
        
        
    def int_dynamic(self, winobj):
        if self.btn_dynamic.text() == 'int. w. watchdog (Ctrl+1)':
        # if not self.process_started:
        #     self.process_started = True
            # self.manager = multiprocessing.Manager()
            # self.manager_list = self.manager.list([])
            self.manager_list = [] # to do: populate with what is already there...
            event_handler = NewFileHandler(self)
            self.observer = PollingObserver()
            rawfolder_list = self.rawfilename.split('\\')
            rawfolder = rawfolder_list[0]
            for rf in rawfolder_list[1:-1]:
                rawfolder += '\\' + rf
                
            self.observer.schedule(event_handler, rawfolder, recursive=False)
            print(rawfolder)
            self.observer.start()
            print('watchdog started...')
            # self.int_workers = []
            # self.int_workers.append(multiprocessing.Process(target=self.integration_worker,
                                                            # args=(self.manager_list,)))
            # self.int_workers[-1].start()
            
            # to do: maybe put somewhere else
            if not self.checksdict['time series'].isChecked():
                self.checksdict['time series'].setChecked(True)
            
            if not self.curvedict['time series']['pointer'].isChecked():
                self.curvedict['time series']['pointer'].setChecked(True)
                
            # pw = winobj.gdockdict[self.slider.objectName()].tabdict['time series'].tabplot
            # pr = int(self.parameters['time series']['plot range'].setvalue)
            if not hasattr(self, 'ts_list'):
                self.ts_list = {} # may put somewhere else
            #     self.ts_text = [] # this has caused the fatal error: Windows access violation
            
            # for i in range(pr):
            #     self.ts_list.append(pw.plot(name=f'ts_curve_{i}'))
                # self.ts_text.append(pg.TextItem(''))
                # pw.addItem(self.ts_text[-1])
                # self.ts_text[-1].setPos(i/10,i/10)
                
            # it's pity that if you setPos(0,0), the graph will go crazy!!!
            # pw.setXRange(0, 1)
            # pw.setYRange(0, 1)
            
            self.worker = Worker(self.integration_worker, winobj) 
            # self.worker = Worker(self.integration_worker(winobj)) 
            # this will result in TyperError: 'NoneType' object is not callable
            self.worker.signals.finished.connect(self.thread_complete)
            winobj.threadpool.start(self.worker)
            print('int. started...')
            self.btn_dynamic.setText('stop int. & watchdog (Ctrl+0)')
            self.btn_dynamic.setShortcut('Ctrl+0')
            
        # if self.process_started:
        #     self.process_started = False
            # the while loop stops the integration intrinsically
        else:
            self.observer.stop()
            print('watchdog stopped')
            self.btn_dynamic.setText('int. w. watchdog (Ctrl+1)')
            self.btn_dynamic.setShortcut('Ctrl+1')
    
    
    def thread_complete(self):
        print('the thread stopped')
    
    
    def data_scale_ts(self, x_axis, scale, data):
        if x_axis == 'q': x = data[0,:]
        if x_axis == '2th': x = np.arcsin(data[0,:] * self.wavelength / 4 / np.pi) * 2 / np.pi * 180
        if x_axis == 'd': x = np.pi * 2 / data[0,:]
        if scale == 'log10': y = np.log10(data[1,:])        
        if scale == 'sqrt': y = np.sqrt(data[1,:])
        if scale == 'linear': y = data[1,:]
        return np.array([x, y])
        
    
    def integration_worker(self, winobj):
        # while len(self.manager_list) > 0: # this does not work when the folder is empty
        # to do: creat folder if not exist
        # while self.process_started:
        pts = self.parameters['time series']
        pw = winobj.gdockdict[self.slider.objectName()].tabdict['time series'].tabplot
        ps = float(pts['plot spacing'].setvalue)
        pr = int(pts['plot range'].setvalue)
        while self.btn_dynamic.text() == 'stop int. & watchdog (Ctrl+0)':
            if len(self.manager_list) > 0:
                print(' working on ' + self.manager_list[0])
                sub_dir = self.intfile.split('\\')
                out_file = os.path.join(sub_dir[0], os.sep, sub_dir[1])
                for sd in sub_dir[2:-1]:
                    out_file = os.path.join(out_file, sd)    
                    if not os.path.isdir(out_file):
                        os.mkdir(out_file)
                        print('mkdir: ' + out_file)
                
                out_file = os.path.join(out_file, 
                                        self.manager_list[0].split('\\')[-1].split('.')[0] + '.'
                                        + self.intfile.split('\\')[-1].split('.')[-1])
                print('aim to output as: ' + out_file)
                
                int_success, q, I = self.int_pyfai(self.manager_list[0], out_file)
                
                if int_success:
                    self.ts_list[out_file] = pw.plot(name=out_file.split('\\')[-1])
                    self.int_data[out_file] = np.array([q, I])
                    self.manager_list.pop(0)
                    L = len(self.int_data) - 1
                    if L > 0: # set range to 0,0 may not work
                        self.slider.setRange(0, L)
                        self.slider.setValue(L)
                    
                    # update time series
                    if pts['content'].choice == 'int.':
                        if pts['plot style'].choice == 'waterfall':
                            # method 1, add new data according to data number
                            data = self.data_scale_ts(pts['x axis'].choice, 
                                                      pts['scale'].choice, 
                                                      self.int_data[out_file])
                            i = out_file.split('.')[-2].split('-')[-1]
                            if i.isnumeric():
                                i = int(i)
                                pen = pg.mkPen(pg.intColor(i * self.huestep, self.maxhue), width=1.5)
                                self.ts_list[out_file].setData(data[0, :], data[1, :] + i * ps, pen=pen)
                                pw.setYRange(data[1,:].min() - ps * pr + i * ps, 
                                             data[1,:].max() + i * ps)
                            else:
                                print('the name does not have a regular format to get the series number')
                            
                            # method 2, repeatedly erase and set new data for curves
                            # data_list = sorted(self.int_data.keys())
                            # for i in range(len(data_list)): # repeatedly erase and set new data for curves
                            #     data = self.data_scale_ts(pts['x axis'].choice, 
                            #                               pts['scale'].choice, 
                            #                               self.int_data[data_list[-1 - i]])
                            #     pen = pg.mkPen(pg.intColor(i * self.huestep, self.maxhue), width=1.5)
                            #     self.ts_list[data_list[-1 - i]].setData(data[0, :], data[1, :] - i * ps, pen=pen)
                                
                            # pw.setYRange(data[1,:].min() - ps * pr, data[1,:].max()) # work?
                                
                    # self.manager_list.pop(0)
                    

    def int_pyfai(self, data_file, out_file):
        if hasattr(self, 'ai'):           
            if self.mask.split('.')[-1] in ['npy']: # to do: more scenario
                mask_file = np.load(self.mask)
            elif self.mask.split('.')[-1] in ['tif','edf']: # to do: more types?
                mask_file = fabio.open(self.mask).data
            else:
                print('can not read in mask, int. w/o mask')
            
            # if 1: # overwrite mode? or not able to intrinsically
            if not os.path.isfile(out_file): # not overwrite 
                print(f'output file as {out_file}')
                q, I = self.ai.integrate1d( # to do: collect q, I
                    fabio.open(data_file).data, 
                    self.int_bins, 
                    filename=out_file, 
                    polarization_factor=0.99, 
                    mask=mask_file,
                    correctSolidAngle=True, 
                    unit='q_A^-1', 
                    method='nosplit_csr',
                    )
                
                return True, q, I
            
        else:
            return False, None, None
        

    def int_static(self, winobj):
        pass
        # for rawfile in sorted(glob.glob(self.rawfilename)):
        #     if os.path.isfile(rawfile):
        #         int_success = self.int_pyfai(self.manager_list[-1])
        #         if int_success:
        #             print('one file done')


class NewFileHandler(FileSystemEventHandler):
    def __init__(self, methodobj):
        super(NewFileHandler, self).__init__()
        self.methodobj = methodobj
    
    # on_modified => file edited
    # on_created  => file created

    def on_modified(self, event):
        pass

    def on_created(self, event):
        # to do: either glob or regex
        # if re.match(self.methodobj.rawfilename.split('\\')[-1],
        #             event.src_path.split('\\')[-1]):
        if event.src_path in glob.glob(self.methodobj.rawfilename):
            self.methodobj.manager_list.append(event.src_path)
            print(event.src_path + ' found by watchdog')
            # prevent warning: invalid value encountered in log10
            data = fabio.open(event.src_path).data
            data[data < 0] = data[data > 0].min()
            self.methodobj.raw_data[event.src_path] = data
            
    


class TOT_general(Methods_Base):
    def __init__(self, path_name_widget):
        super(TOT_general, self).__init__()
        self.availablemethods = ['raw', 'integrated', 'F(Q)', 'G(r)', 'time series']
        self.availablecurves['raw'] = ['image', 'mask', 'gain map', 'dark'] # to do: no gain map yet
        self.availablecurves['integrated'] = ['original', 'bkg']
        self.availablecurves['time series'] = ['pointer']
        self.availablecurves['S(Q)'] = ['Sm(Q)', '<font> I(Q)/&lt; f &gt; </font> <sup> 2 </sup>', 
                                        '</font> &lt; f &gt; <sup> 2 </sup> /&lt; f &gt; </font> <sup> 2 </sup>']
        self.availablecurves['F(Q)'] = ['Fm(Q)', 'QPn(Q)']
        self.availablecurves['G(r)'] = ['g(r)', 'G(r)', 'R(r)', 'N(r)']
        self.directory = path_name_widget['directory'].text()
        self.rawfilename = self.directory + path_name_widget['raw pattern'].text() # to do: there is no fileshort now
        self.dark = self.directory + path_name_widget['dark file'].text() # to do: can be only one, or one for each image
        self.mask = self.directory + path_name_widget['mask file'].text() # to do: may be several
        self.config = self.directory + path_name_widget['config file'].text() # to do: may not supply
        self.intfile = self.directory + path_name_widget['int. pattern'].text()
        # self.intfile_appendix = path_name_widget['result appendix'].text() # to do: this may only serve different way of int.
        self.ponifile = self.directory + path_name_widget['PONI file'].text()
        self.int_bins = int(path_name_widget['int. bins'].text())
        print(self.ponifile)
        if os.path.isfile(self.ponifile): # to do: is this step necessary
            # self.ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator()
            # module 'pyFAI' has no attribute 'azimuthalIntegrator'
            self.ai = pyFAI.AzimuthalIntegrator()
            self.ai.load(self.ponifile)
            with open(self.ponifile, 'r') as f:
                self.wavelength = 1e10 * float(f.readlines()[-1].splitlines()[0].partition(' ')[2]) # now in Angstrom
                print(f'wavelength read in as {self.wavelength} Angstrom')
        else: 
            print('no poni file, wavelength not known')
        
        
        # to do: this sort could be taken over by watchdog
        raw_list = sorted(glob.glob(self.rawfilename))
        if raw_list: # is not empty
            self.slider.setRange(0, len(raw_list) - 1)
        else:
            self.slider.setRange(0, 1)
            
        self.slider.setSingleStep(1)
        
        self.axislabel = {'raw':{'bottom':'',
                                 'left':''},
                          'integrated': {'bottom': '<font> q / &#8491; </font> <sup> -1 </sup>,'
                                                   '<font> 2 &#952; / </font> <sup> o </sup>, or'
                                                   '<font> d / &#8491; </font>',
                                        'left': 'Intensity'},
                          'time series': {'bottom': '<font> q / &#8491; </font> <sup> -1 </sup>,'
                                                   '<font> 2 &#952; / </font> <sup> o </sup>, or'
                                                   '<font> d / &#8491; </font>',
                                        'left': '<-- Data number --'},
                          # 'time series': {'bottom': '<font> r / &#8491; </font>',
                          #               'left': 'Data number'},
                          'S(Q)': {'bottom': '<font> q / &#8491; </font> <sup> -1 </sup>',
                                   'left': 'Intensity'},
                          'F(Q)': {'bottom': '<font> q / &#8491; </font> <sup> -1 </sup>',
                                   'left': 'Intensity'},
                          'G(r)': {'bottom': '<font> r / &#8491; </font>',
                                   'left': 'Intensity'},
                          }

        self.ini_data_curve_color()

        self.parameters = {'integrated': {'scale': Paraclass(strings=('log10', ['log10', 'sqrt', 'linear'])),
                                          'x axis': Paraclass(strings=('q',['q','2th','d'])),
                                          },
                           'time series': {'scale': Paraclass(strings=('log10', ['log10', 'sqrt', 'linear'])),
                                           'x axis': Paraclass(strings=('q',['q','2th','d'])),
                                           'plot style': Paraclass(strings=('waterfall',['waterfall','heatmap'])),
                                           'content':Paraclass(strings=('int.',['int.','F(Q)','G(r)'])),
                                           'plot range': Paraclass(values=(1,1,int(self.maxhue / self.huestep),1)),
                                           'plot spacing': Paraclass(values=(.1,0,int(self.maxhue / self.huestep),.1)),
                                           },
                           }
        
        self.actions = {'time series':{'update plot (Ctrl+U)':self.update_ts,
                                       },
                        }
        
        self.linedit = {'raw':{'color max':'5'
                               },
                        }

    def update_ts(self):
        pass
        
    
    
    def data_scale(self, mode, sub_mode, data_x, data_y):  # for data_process
        if self.parameters[mode]['x axis'].choice == 'q':
            data_x = data_x
        if self.parameters[mode]['x axis'].choice == '2th':
            # with open(self.ponifile, 'r') as f:
            #     wavelength = float(f.readlines()[-1].splitlines()[0].partition(' ')[2]) * 1e10  # in Angstrom
            data_x = np.arcsin(data_x * self.wavelength / 4 / np.pi) * 2 / np.pi * 180
        if self.parameters[mode]['x axis'].choice == 'd':
            data_x = 2 * np.pi / data_x
        if self.parameters[mode]['scale'].choice == 'log10':
            self.data_timelist[0][mode][sub_mode].data = np.transpose([data_x, np.log10(np.abs(data_y))])
        if self.parameters[mode]['scale'].choice == 'sqrt':
            self.data_timelist[0][mode][sub_mode].data = np.transpose([data_x, np.sqrt(data_y)])
        if self.parameters[mode]['scale'].choice == 'linear':
            self.data_timelist[0][mode][sub_mode].data = np.transpose([data_x, data_y])


    def data_process(self, para_update): # for curves, embody data_timelist, if that curve exists
        # energy/wavelength can be acquired from poni file
        # self.read_data_time()  # gives raw and integrated data depending on checks to do: maybe for time series

        # raw
        if 'image' in self.curve_timelist[0]['raw']:
            self.colormax = float(self.linewidgets['raw']['color max'].text())
            rawfiles = sorted(glob.glob(self.rawfilename)) # to do : alternatinve: self.raw_data
            if len(rawfiles) >= self.slider_v: # not empty
            # if len(self.raw_data) >= self.slider_v: # not empty
                self.dynamictitle = rawfiles[self.slider_v].split('\\')[-1]
                self.data_timelist[0]['raw']['image'].image = \
                    pg.ImageItem(image=np.log10(self.raw_data[rawfiles[self.slider_v]])) # faster?
                    
                    # pg.ImageItem(image=np.log10(fabio.open(rawfiles[self.slider_v]).data)) # to do not able to open?
                    
                    # pg.ImageItem(image=np.log10(tifffile.imread(rawfiles[self.slider_v])))
                    
                
                # print('image loaded')

        # integrated
        if 'original' in self.curve_timelist[0]['integrated']:
            intfiles = sorted(glob.glob(self.intfile)) # to do : alternatinve: self.int_data
            if len(intfiles) >= self.slider_v: # two occasions: q/A^-1 or 2th_deg
            # if len(self.int_data) >= self.slider_v: # not empty
                self.dynamictitle = intfiles[self.slider_v].split('\\')[-1]
                intdata = np.loadtxt(intfiles[self.slider_v])
                d_x = []
                with open(intfiles[self.slider_v]) as f:
                    for line in f:
                        if line[0] == '#' and len(line) >= 9:
                            if line[8:15] in [' q_A^-1', '2th_deg']: # for pyFAI int.
                                if line[8:17] in ['q_A^-1 ']:
                                    d_x = intdata[:,0]
                                else:
                                    d_x = np.sin(intdata[:,0] / 180 * np.pi / 2) * 4 * np.pi / self.wavelength
                                
                if len(d_x) > 0:
                    # print('plot' + line[1:])
                    self.data_scale('integrated', 'original', d_x, intdata[:,1])

        # time series
        if 'pointer' in self.curve_timelist[0]['time series']:
            intfiles = sorted(self.int_data.keys()) # to do: what if the process is interrupted and the slider value is larger
            if len(intfiles) >= self.slider_v: # two occasions: q/A^-1 or 2th_deg
                self.dynamictitle = intfiles[self.slider_v].split('\\')[-1]
                # intdata = np.loadtxt(intfiles[self.slider_v])
            
            pts = self.parameters['time series']
            ps = float(pts['plot spacing'].setvalue)
            if pts['content'].choice == 'int.':
                if pts['plot style'].choice == 'waterfall':
                    data = self.data_scale_ts(pts['x axis'].choice, 
                                              pts['scale'].choice, 
                                              self.int_data[intfiles[self.slider_v]])
                    dts = self.data_timelist[0]['time series']['pointer']
                    # dts.data = np.array([data[0,:], 
                    #                      data[1,:] - (len(intfiles) - self.slider_v) * ps]).transpose()
                    # method 1, above method 2
                    i = self.dynamictitle.split('.')[-2].split('-')[-1]
                    if i.isnumeric():
                        dts.data = np.array([data[0,:], 
                                             data[1,:] + int(i) * ps]).transpose()
                        dts.pen = None
                        dts.symbolsize = 10
                        dts.symbolBrush = 'k'
                        dts.symbol = '+'



class XRD_general():
    pass



class ShowData(QMainWindow):
    def __init__(self):
        super(ShowData, self).__init__()
        self.threadpool = QThreadPool() # all methods share this threadpool
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        myscreen = app.primaryScreen()
        self.screen_height = myscreen.geometry().height()
        self.screen_width = myscreen.geometry().width()
        self.setGeometry(0, int(self.screen_height * .1), int(self.screen_width),
                         int(self.screen_height * .85))

        
        self.initialized = False
        
        # self.setCentralWidget() ??
        
        self.controls = QDockWidget('Parameters', self)
        self.controls.setMaximumWidth(int(self.screen_width * .3))
        self.controls.setMinimumWidth(int(self.screen_width * .1))

        self.controltabs = DraggableTabWidget()

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('leftButtonPan', False)

        self.cboxes = QToolBox()
        self.cboxes.setMinimumHeight(int(self.screen_height)) # hope this can make it a bit more spacious
        self.scroll_area = QScrollArea() #
        self.scroll_area.setWidget(self.cboxes) #
        self.scroll_area.setWidgetResizable(True) # this line is the key, OMG!!!
        self.controltabs.addTab(self.scroll_area, "Checkboxes") #
        
        self.controls.setWidget(self.controltabs)
        self.controls.setFloating(False)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.controls) # ?

        self.gdockdict = dict() # a dic for all top-level graph dock widgets: xas, xrd,...
        self.methodict = dict() # a dic for all available methods: xas, xrd,... there should be only one type each!
        self.data_read_dict = {
                               'tot': TOT_general,
                               'xrd': XRD_general,
                               # 'sas': SAS_general,
                               }

        data_read = QMenu('New', self)
        read_mode_group = QActionGroup(data_read)
        for read_mode in self.data_read_dict:
            action = QAction(read_mode,data_read,checkable=True)
            data_read.addAction(action)
            read_mode_group.addAction(action)

        read_mode_group.setExclusive(True)
        # read_mode_group.triggered.connect(self.choose_data_read_mode)

        bar = self.menuBar()
        bar.addMenu(data_read)

        self.checkmethods = {} # checkboxes of top level
        self.cboxwidget = {}
        self.subcboxverti = {}
        self.path_name_dict = {}
        self.path_name_widget = {}
        self.methodclassdict = {}  # a dic for preparing corresponding Class

        read_mode_group.triggered.connect(self.ini_methods_cboxes)

        self.pn_dict = {}
        self.pn_dict['xrd'] = {'directory':r'T:\current', 
                          'raw pattern':r'\raw\pe\Jiatu_test\test*[0-9].raw.tif',
                          'dark file':r'\raw\combined_3_xrd*[0-9].dark.tif',
                          'int. pattern':r'\processed\pe\Jiatu_test\test*[0-9].dat',
                          'mask file':r'\processed\mask_2p50std.npy',
                          'PONI file':r'\processed\Ni_1mm-00002.poni',
                          'int. bins':'1400',
                          'config file':r'.cfg',
                          }
        
        self.pn_dict['tot'] = {'directory':r'D:\test_folder', 
                          'raw pattern':r'\raw\combined_3_xrd*[0-9].tif',
                          'dark file':r'\raw\combined_3_xrd*[0-9].dark.tif',
                          'int. pattern':r'\processed\combined_3_xrd*[0-9].dat',
                          'mask file':r'\processed\scripts\new_mask_2.edf',
                          'PONI file':r'\processed\scripts\LaB6_1_new_m2.poni',
                          'int. bins':'1400',
                          'config file':r'.cfg',
                          }
        
        # self.pn_dict['xrd'] = {'directory':r'Z:\p21.1\2023\data\11017043', 
        #                   'raw pattern':r'\raw\pe\RHEA_16_575\combined_3_ts*[0-9].tif',
        #                   'dark file':r'\raw\pe\RHEA_16_575\combined_3_ts*[0-9].dark.tif',
        #                   'int. pattern':r'\processed\RHEA_16_575_integrated\combined_3_ts*[0-9].dat',
        #                   'mask file':r'\processed\scripts\new_mask_2.edf',
        #                   'PONI file':r'\processed\scripts\LaB6_1_new_m2.poni',
        #                   'int. bins':'1400',
        #                   'config file':r'\shared\xPDFsuite\xPDFsuite_RHEA_16_575.cfg',
        #                   }
    
    
    def ini_methods_cboxes(self, action):
        # add sequentially names: e.g. xrd_, tot_, sas_ to path_name_dict and methodclassdict
        action_name_list = []
        if self.path_name_dict:
            for kn in self.path_name_dict.keys():
                if kn:
                    if kn[:3] == action.text():
                        action_name_list.append(kn)
                
        if action_name_list:
            # n = int(action_name_list[-1].split('_')[-1]) + 1
            # key = f'{action.text()}_{n}'
            key = f'{action.text()}_{len(action_name_list)}'
        else:
            key = f'{action.text()}_0'
        
        self.path_name_dict[key] = {}
        self.methodclassdict[key] = self.data_read_dict[action.text()]
        for kn in self.pn_dict[action.text()]:
            self.path_name_dict[key][kn] = self.pn_dict[action.text()][kn]
        
        self.checkmethods[key] = QCheckBox(key)
        self.checkmethods[key].stateChanged.connect(self.graphdock)
        self.cboxwidget[key] = QWidget()
        self.cboxwidget[key].setObjectName(key)  # important for .parent recognition
        cboxverti = QVBoxLayout()
        cboxfiles = QFormLayout()
        self.path_name_widget[key] = {}
        
        for subkey in self.path_name_dict[key]:
            self.path_name_widget[key][subkey] = QLineEdit(self.path_name_dict[key][subkey])
            self.path_name_widget[key][subkey].setPlaceholderText(subkey)
            cboxfiles.addRow(subkey, self.path_name_widget[key][subkey])
        
        
        cboxverti.addLayout(cboxfiles)
        cboxverti.addWidget(self.checkmethods[key])
        self.subcboxverti[key] = QVBoxLayout()  # where all tab graph checkboxes reside, e.g. raw, norm, ...
        cboxhori = QHBoxLayout()
        cboxhori.addSpacing(10)
        cboxhori.addLayout(self.subcboxverti[key])
        cboxverti.addLayout(cboxhori)
        cboxverti.addStretch(1)
        self.cboxwidget[key].setLayout(cboxverti)
        self.cboxes.addItem(self.cboxwidget[key], key)


    def graphdock(self, state):
        checkbox = self.sender() # e.g. xas, xrd,...
        if state == Qt.Checked:
            # create a method object, e.g. an XAS object
            self.methodict[checkbox.text()] = self.methodclassdict[checkbox.text()](self.path_name_widget[checkbox.text()])
            self.methodict[checkbox.text()].tabchecks(self, checkbox.text()) # show all checkboxes, e.g. raw, norm,...
            # self.slider.valueChanged.connect(self.methodict[checkbox.text()].update_timepoints)
            self.gdockdict[checkbox.text()] = DockGraph(checkbox.text())
            self.gdockdict[checkbox.text()].gendock(self, self.methodict[checkbox.text()])
            self.gdockdict[checkbox.text()].gencontroltab(self)
        else:
            # add someting to prevent the problem mentioned at the end of Classes
            if self.gdockdict[checkbox.text()]:
                self.methodict[checkbox.text()].deltabchecks(self, checkbox.text())
                self.gdockdict[checkbox.text()].deldock(self)
                self.gdockdict[checkbox.text()].delcontroltab(self)
                if checkbox.text()[0:3] == 'xrd' and hasattr(self.methodict[checkbox.text()], 'index_win'): # for index window deletion
                    self.methodict[checkbox.text()].index_win.deldock(self)
                # self.slider.valueChanged.disconnect(self.methodict[checkbox.text()].update_timepoints)
                del self.methodict[checkbox.text()] # del the method object
                print('del method ' + checkbox.text())
                del self.gdockdict[checkbox.text()] # del the dock object, here can do some expansion to prevent a problem mentioned in Classes
                print('del dock ' + checkbox.text())

    def graph_tab(self, state): # to do: may be better to put it in DockGraph?
        checkbox = self.sender() # e.g. raw, norm,...
        tooltabname = checkbox.parent().objectName() # e.g. xas, xrd,...
        if state == Qt.Checked:
            self.gdockdict[tooltabname].tabdict[checkbox.text()] = TabGraph(checkbox.text())
            tabgraphobj = self.gdockdict[tooltabname].tabdict[checkbox.text()]
            tabgraphobj.gentab(self.gdockdict[tooltabname], self.methodict[tooltabname])
            tabgraphobj.gencontrolitem(self.gdockdict[tooltabname])
            tabgraphobj.curvechecks(checkbox.text(), self.methodict[tooltabname], self)
        else:
            if self.gdockdict[tooltabname].tabdict[checkbox.text()]:
                tabgraphobj = self.gdockdict[tooltabname].tabdict[checkbox.text()]
                tabgraphobj.delcurvechecks(checkbox.text(), self.methodict[tooltabname])
                print('del 1 ' + checkbox.text())
                tabgraphobj.deltab(self.gdockdict[tooltabname])
                print('del 2 ' + tooltabname)
                tabgraphobj.delcontrolitem(self.gdockdict[tooltabname])
                print('del 3 ' + tooltabname)
                del self.gdockdict[tooltabname].tabdict[checkbox.text()] # access violation?
                print('del 4 ' + checkbox.text())
    
    def graphcurve(self, state):
        checkbox = self.sender() # e.g. I0, I1,...
        toolitemname = checkbox.parent().objectName() # e.g. raw, norm,...
        tooltabname = checkbox.parent().accessibleName() # e.g. xas, xrd,...
        if state == Qt.Checked:
            for timelist in range(len(self.methodict[tooltabname].curve_timelist)):  # timelist: 0, 1,...
                # a curve obj is assigned to the curve dict
                self.methodict[tooltabname].curve_timelist[timelist][toolitemname][checkbox.text()] = \
                    self.gdockdict[tooltabname].tabdict[toolitemname].tabplot.plot(name=checkbox.text())
        else:
            for timelist in range(len(self.methodict[tooltabname].curve_timelist)):  # timelist: 0, 1,...
                if checkbox.text() in self.methodict[tooltabname].curve_timelist[timelist][toolitemname]:
                    self.gdockdict[tooltabname].tabdict[toolitemname].tabplot.removeItem(
                        self.methodict[tooltabname].curve_timelist[timelist][toolitemname][checkbox.text()]
                    )
                    del self.methodict[tooltabname].curve_timelist[timelist][toolitemname][checkbox.text()]
                    print('del curve in method ' + tooltabname + ' ' + checkbox.text())
                if checkbox.text() in self.methodict[tooltabname].data_timelist[0][toolitemname]:
                    if self.methodict[tooltabname].data_timelist[0][toolitemname][checkbox.text()].image != None:
                        self.gdockdict[tooltabname].tabdict[toolitemname].tabplot.clear()


    def memorize_curve(self):
        # memorize the slider value for re-load
        self.slidervalues.append(self.slider.value()) # need to divided by 1000 ???
        for key in self.gdockdict: # key as xas, xrd,...
            # prepare the data, curve
            data_mem, curve_mem = self.methodict[key].data_curve_copy(self.methodict[key].data_timelist[0])
            self.methodict[key].data_timelist.append(data_mem)
            self.methodict[key].curve_timelist.append(curve_mem)
            # plot the curve onto corresponding tabgraph
            for subkey in self.methodict[key].curve_timelist[0]: # subkey as raw, norm,...
                if subkey != 'refinement single':
                    for entry in self.methodict[key].curve_timelist[0][subkey]: # entry as I0, I1,...
                        # this is for occasions when there is only .image
                        # and not including truncated, peaks in xrd
                        if data_mem[subkey][entry].data is not None and \
                                entry not in ['find peaks', 'truncated']:
                            curve_mem[subkey][entry] = \
                                self.gdockdict[key].tabdict[subkey].tabplot.plot(name=self.methodict[key].dynamictitle + ' ' + entry)
                            # set data for each curve
                            curve_mem[subkey][entry].setData(data_mem[subkey][entry].data)
                            if data_mem[subkey][entry].pen: curve_mem[subkey][entry].setPen(data_mem[subkey][entry].pen)
                            if data_mem[subkey][entry].symbol: curve_mem[subkey][entry].setSymbol(data_mem[subkey][entry].symbol)
                            if data_mem[subkey][entry].symbol:
                                curve_mem[subkey][entry].setSymbolSize(data_mem[subkey][entry].symbolsize)
                            if data_mem[subkey][entry].symbol:
                                curve_mem[subkey][entry].setSymbolBrush(data_mem[subkey][entry].symbolbrush)


    def clear_curve(self):
        # clear the slider value for re-load
        if self.slidervalues != []:
            self.slidervalues.pop()
        # proof to clear the first obj ?
        for key in self.gdockdict:
            if len(self.methodict[key].data_timelist) > 1:
                # tune back the colorhue
                for subkey in self.methodict[key].availablemethods:
                    for entry in self.methodict[key].availablecurves[subkey]:
                        self.methodict[key].colorhue[subkey] -= self.methodict[key].huestep
                # del individual curves
                for subkey in self.methodict[key].curve_timelist[-1]:  # subkey as raw, norm,...
                    if len(self.methodict[key].curve_timelist[-1][subkey]) > 0:
                        for entry in self.methodict[key].curve_timelist[-1][subkey]:  # entry as I0, I1,...
                            self.gdockdict[key].tabdict[subkey].tabplot.removeItem(
                                self.methodict[key].curve_timelist[-1][subkey][entry]
                            )
                # del curves
                del self.methodict[key].curve_timelist[-1]
                print('del curve in method ' + key)
                # del data
                del self.methodict[key].data_timelist[-1]
                print('del data in method ' + key)



    def update_timepoints(self, slidervalue): # slidervalue in ms !
        key = self.sender().objectName() # tot_, xrd_
        # print(f'slidervalue {slidervalue}')
        self.methodict[key].sliderlabel.setText(str(slidervalue))
        try:
            self.methodict[key].data_update(slidervalue)
        except Exception as exc:
            print(exc)
            print(f'can not update data in {key}')

        if self.methodict[key].update == True:
            # print(f'{key} updating')
            try:
                self.update_curves(key)
            except Exception as exc:
                print(exc)
                print(f'can not update curves')
                

    def update_parameters(self, widgetvalue):
        # rewrite this part!!!
        if self.slideradded == False: # a way to prevent crash; may not the best way
            self.setslider()
            self.slider.setValue(self.slider.minimum() + 1) # this will not work!!!
            self.slideradded = True

        parawidget = self.sender() # rbkg, kmin,...
        toolitemname = parawidget.parent().objectName() # post edge, chi(k),...
        tooltabname = parawidget.parent().accessibleName() # xas, xrd,...
        temppara = self.methodict[tooltabname].parameters[toolitemname][parawidget.objectName()]
        if type(widgetvalue) == str:
            temppara.choice = widgetvalue
        else:
            nominal_value = widgetvalue * temppara.step + temppara.lower
            self.methodict[tooltabname].paralabel[toolitemname][parawidget.objectName()].setText(
                parawidget.objectName() + ':{:.1f}'.format(nominal_value))
            temppara.setvalue = nominal_value

        if tooltabname[0:3] == 'xrd' and toolitemname == 'time series':

            if parawidget.objectName() == 'scale':
                self.methodict[tooltabname].plot_from_load(self)

            elif parawidget.objectName() == 'normalization':
                self.methodict[tooltabname].plot_from_load(self)

            # elif parawidget.objectName() in ['gap y tol.', 'gap x tol.', 'min time span', '1st deriv control']:
                # self.methodict[tooltabname].catalog_peaks(self)

            # elif parawidget.objectName() in ['max diff start time', 'max diff time span']:
            #     self.methodict[tooltabname].assign_phases(self)

            elif parawidget.objectName() == 'single peak int. width':
                self.methodict[tooltabname].single_peak_width(self)

            elif parawidget.objectName() == 'symbol size':
                self.methodict[tooltabname].peak_map.setSymbolSize(int(nominal_value))
                if hasattr(self.methodict[tooltabname],'peaks_catalog') and \
                        self.methodict[tooltabname].peaks_catalog_map != []:
                    for index in range(len(self.methodict[tooltabname].peaks_catalog_map)):
                        self.methodict[tooltabname].peaks_catalog_map[index].setSymbolSize(int(nominal_value))

                if hasattr(self.methodict[tooltabname],'phases') and \
                        self.methodict[tooltabname].phases_map != []:
                    for index in range(len(self.methodict[tooltabname].phases_map)):
                        self.methodict[tooltabname].phases_map[index].setSymbolSize(int(nominal_value))

            elif parawidget.objectName() == 'phases':
                self.methodict[tooltabname].show_phase(self)

        else:
            # if tooltabname[0:3] == 'xas' and (toolitemname in ['normalizing', 'normalized']) and \
            #         parawidget.objectName()[0:8] == 'Savitzky':
            #     self.methodict[tooltabname].filtered = True

            self.methodict[tooltabname].data_process(True) # True means the parameters have changed
            self.update_curves(tooltabname)

    def update_curves(self, key):
        for subkey in self.gdockdict[key].tabdict: # raw, norm,...
            if self.methodict[key].dynamictitle:
                self.gdockdict[key].tabdict[subkey].tabplot.setTitle(self.methodict[key].dynamictitle)
                
            if subkey == 'refinement single':
                self.gdockdict[key].tabdict[subkey].tabplot.setTitle(
                    self.methodict[key].refinedata[self.methodict[key].parameters[subkey]['data number'].setvalue])
            
            if subkey == 'time series': # only update view range
                pts = self.methodict[key].parameters['time series']
                if pts['content'].choice == 'int.':
                    if pts['plot style'].choice == 'waterfall':
                        pw = self.gdockdict[key].tabdict['time series'].tabplot
                        data_list = sorted(self.methodict[key].int_data.keys())
                        sv = self.methodict[key].slider_v
                        ps = float(pts['plot spacing'].setvalue)
                        pr = int(pts['plot range'].setvalue)
                        data = self.methodict[key].data_scale_ts(pts['x axis'].choice, 
                                                                 pts['scale'].choice, 
                                                                 self.methodict[key].int_data[data_list[sv]])
                        # method 2
                        # pw.setYRange(data[1,:].min() - (pr + len(data_list) - sv) * ps, 
                        #              data[1,:].max() - (pr + len(data_list) - sv) * ps)
                        # method 1
                        pw.setYRange(data[1,:].min() + (sv - pr) * ps, 
                                     data[1,:].max() + sv * ps)
            
            for timelist in range(len(self.methodict[key].curve_timelist)): # timelist: 0, 1,...
                for entry in self.methodict[key].curve_timelist[timelist][subkey]: # I0, I1,...
                    curveobj = self.methodict[key].curve_timelist[timelist][subkey][entry]
                    if entry in self.methodict[key].data_timelist[timelist][subkey]:
                        dataobj = self.methodict[key].data_timelist[timelist][subkey][entry]
                        if dataobj.data is not None: # now in spyder we use is not... dataobj.data != None:
                            curveobj.setData(dataobj.data, pen=dataobj.pen, symbol=dataobj.symbol,
                                             symbolSize=dataobj.symbolsize, symbolBrush=dataobj.symbolbrush)
                        if dataobj.image is not None: # for xrd raw image
                            if hasattr(self.methodict[key],'color_bar_raw'):
                                self.gdockdict[key].tabdict[subkey].tabplot.clear() # may not be the best if you want to process raw onsite
                                self.methodict[key].color_bar_raw.close()

                            self.gdockdict[key].tabdict[subkey].tabplot.setAspectLocked()
                            self.gdockdict[key].tabdict[subkey].tabplot.addItem(dataobj.image)
                            self.methodict[key].color_bar_raw = \
                                pg.ColorBarItem(values=(0, self.methodict[key].colormax), colorMap=pg.colormap.get('CET-L3')) 
                                # cmap keyword is no longer supported
                            self.methodict[key].color_bar_raw.setImageItem(dataobj.image,
                                                                       insert_in=self.gdockdict[key].tabdict[subkey].tabplot)




if __name__ == '__main__':
    # solve the dpi issue
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    faulthandler.enable() #start @ the beginning
    make_dpi_aware()
    app = QApplication(sys.argv)
    w = ShowData()
    w.show()
    sys.exit(app.exec_())


