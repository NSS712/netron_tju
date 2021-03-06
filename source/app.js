/* eslint-disable brace-style */
/* eslint-disable semi */
/* eslint-disable no-trailing-spaces */
/* eslint-disable no-irregular-whitespace */
/* eslint-disable no-unused-vars */
/* eslint-disable prefer-const */
/* jshint esversion: 6 */

const { timerFlush } = require('d3-timer');
const electron = require('electron');
const updater = require('electron-updater');
const fs = require('fs');
const os = require('os');
const path = require('path');
const process = require('process');
const url = require('url');

class Application {

    constructor() {

        this._views = new ViewCollection();
        this._configuration = new ConfigurationService();
        this._menu = new MenuService();
        this._openFileQueue = [];

        electron.app.setAppUserModelId('com.lutzroeder.netron');
        electron.app.allowRendererProcessReuse = true;

        if (!electron.app.requestSingleInstanceLock()) {
            electron.app.quit();
            return;
        }

        electron.app.on('second-instance', (event, commandLine, workingDirectory) => {
            const currentDirectory = process.cwd();
            process.chdir(workingDirectory);
            const open = this._parseCommandLine(commandLine);
            process.chdir(currentDirectory);
            if (!open) {
                if (this._views.count > 0) {
                    const view = this._views.item(0);
                    if (view) {
                        view.restore();
                    }
                }
            }
        });

        electron.ipcMain.on('open-file-dialog', () => {
            this._openFileDialog();
        });

        electron.ipcMain.on('get-environment', (event) => {
            event.returnValue = {
                version: electron.app.getVersion(),
                package: electron.app.isPackaged,
                zoom: 'd3'
                // zoom: 'scroll'
            };
        });
        electron.ipcMain.on('get-configuration', (event, obj) => {
            event.returnValue = this._configuration.has(obj.name) ? this._configuration.get(obj.name) : undefined;
        });
        electron.ipcMain.on('set-configuration', (event, obj) => {
            this._configuration.set(obj.name, obj.value);
        });
        electron.ipcMain.on('drop-files', (event, data) => {
            const files = data.files.filter((file) => fs.statSync(file).isFile());
            this._dropFiles(event.sender, files);
        });
        electron.ipcMain.on('show-message-box', (event, options) => {
            const owner = event.sender.getOwnerBrowserWindow();
            event.returnValue = electron.dialog.showMessageBoxSync(owner, options);
        });
        electron.ipcMain.on('show-save-dialog', (event, options) => {
            const owner = event.sender.getOwnerBrowserWindow();
            event.returnValue = electron.dialog.showSaveDialogSync(owner, options);
        });


        electron.ipcMain.on('Export TFLite', (event, data) => {
            this.rky_SaveTfliteJson(data);
        })

        electron.app.on('will-finish-launching', () => {
            electron.app.on('open-file', (event, path) => {
                this._openFile(path);
            });
        });

        electron.app.on('ready', () => {
            this._ready();
        });

        electron.app.on('window-all-closed', () => {
            if (process.platform !== 'darwin') {
                electron.app.quit();
            }
        });

        electron.app.on('will-quit', () => {
            this._configuration.save();
        });

        this._parseCommandLine(process.argv);
        this._checkForUpdates();
    }

    rky_SplitChain(node) {
        outputs = node._outputs;
        if (node._chain && node._chain._length > 0) {
            const chainOutputs = node._chain[node._chain._length - 1]._outputs;
            if (chainOutputs._length > 0) {
                outputs = chainOutputs;
            }
        }
        if (outputs._arguments._length != 1) {
            console._error('only support 1 output for chained/fused operators');
            return [node];
        }
        // rocky: todo: ??????chain????????????????????????????????????tensor???????????????nodes
        curOutputs = outputs;
        newNodes = [node];
        for (let i=0; i<node._chain._length; i++) {
        }
        return newNodes;
    }

    rky_IsGraphInput(tns, nodes) {
        if (tns.netronTns._initializer != null) {
            // ??????weights / bias ??????????????????
            return false;
        }
        for (let node of nodes) {
            for (let output of node._outputs) {
                for (let arg of output._arguments) {
                    if (tns.name == arg._name) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    rky_IsGraphOutput(tns, nodes) {
        if (tns.netronTns._initializer != null) {
            // ??????weights / bias ??????????????????
            return false;
        }
        for (let node of nodes) {
            for (let input of node._inputs) {
                for (let arg of input._arguments) {
                    if (tns.name == arg._name) {
                        return false;
                    }
                }
            }
        }
        return true;
    }    

    rky_CheckGroupConv(node, op){
        op._name = 'CONV_2D';    // ??????????????????Conv2D??????????????????
        for (let att of node._attributes) {
            if (att._name == 'group') {
                if (att._value != 1) {
                    // ???????????????1._????????????????????????????????????
                    let ic = 0; // ic = input channels
                    for (let i = 0; i< node._inputs._length; i++) {
                        // ????????????????????????
                        for (let arg of node._inputs[i]._arguments) {
                            // ????????????????????????
                            if (arg._initializer == null){
                                // ??????????????????????????????
                                continue;
                            }                        
                            if (arg._type != null) {
                                // ???????????????????????????????????????CHW?????????
                                ic = arg._type._shape._dimensions[0];
                                break;
                            }
                        }
                        if (ic != 0) {
                            break;
                        }
                    }

                    // rky: ??????bug: ????????????Int64????????????._low
                    if (att._value._low === ic._low) {
                        // ???????????????????????????????????????PyTorch/Caffe/onnx ????????????dsconv?????????
                        op._name = 'DEPTHWISE_CONV_2D';
                    } else {
                        op._name = 'GROUPED_CONV2D'
                    }
                }
                break;
            }
        }
    }

    rky_ParseTensor(tns, netronTns, nodes, tnsAry, isInput) {
        if (netronTns._initializer != null) {
            tns.isConst = true;
            tns.data = {
                // rky: ????????????????????????????????????????????????????????????????????????onnx, caffemodel
                ary: netronTns._initializer._data,
                shape: netronTns._type._shape
            };
            console.log(tns.data._shape);
        } else {
            tns.isConst = false;
            tns.data = null;
        }
                      
        if (isInput == true) {
            tns.isGraphInput = this.rky_IsGraphInput(tns, nodes);
            tns.isGraphOutput = false;
        } else {
            tns.isGraphOutput = this.rky_IsGraphOutput(tns, nodes);
            tns.isGraphInput = false;
        }        
        tnsAry.push(tns);
    }

    _quantization_(nt, t)
    {
        // ??????quantization - llw

            nt.quantization = {};
            let _min;
            let _max;
            let _scale;
            let _zp;
            if (t.netronTns._initializer != null) {
                // ??????_min ??????_max
                _min = t.netronTns._initializer._buffer[0];
                _max = t.netronTns._initializer._buffer[0];

                for (let temp of t.netronTns._initializer._buffer) {
                    if (temp <= _min) {
                        _min = temp;
                    }
                    if (temp >= _max) {
                        _max = temp;
                    }
                }
                // ??????_scale
                _scale = (_max - _min) / 255.0 - 0.0;

                // ??????_zp
                _zp = 255.0 - _max / _scale;
                _zp = _zp.toFixed(0);

            }
            else {
                _zp = 0;
                _scale = 0.000991;
            }


            if (t.netronTns._initializer != null) {
                nt.quantization.min = [];
                nt.quantization.max = [];
                nt.quantization.scale = [];
                nt.quantization.zero_point = [];
                nt.quantization.min.push(_min);
                nt.quantization.max.push(_max);
                nt.quantization.scale.push(_scale);
                nt.quantization.zero_point.push(_zp);
            }
            else {
                nt.quantization.scale = [];
                nt.quantization.zero_point = [];
                nt.quantization.scale.push(_scale);
                nt.quantization.zero_point.push(_zp);
            }
    }

    rky_SaveTfliteJson(view) {
        let sFileName = view._model_folder + view._modelfile + '._json';
        let model = view._model;
        let dct = {
            version: 3,
        }
        let sg_aryTensors = []; // ?????????????????????tensor
        let sg_aryInpTensors = [];  // ?????????????????????tensor
        let sg_aryOutTensors = [];  // ?????????????????????tensor
        let sg_aryOps = [];     // ?????????????????????
        let sg_aryOpTypes = []; // ???????????????????????????, ??????tflite??????OperatorCode
        // ?????????????????? (??????????????????????????????????????????)???
        let sg_sName = view._model_name;
        let sFmt = view._format;
        let isCHW = false;  // ?????????CHW??????????????????????????????HWC?????????
        let typeoftensor = [];//??????tensor??????
        // onnx??????CHW?????????
        if (sFmt.indexOf('onnx') >= 0) {
            isCHW = true;
        }

        if (model._graphs.length != 1) {
            // tflite?????????????????????
            console.log('sub graph must be 1!');
            return;
        }
        let nodeNdx = 0;
        let tnsNdx = 0;
        let nodes = model._graphs[0]._nodes; // rky: netron???node???operator
        for (let node of nodes) {
            // ?????????node (?????????????????????????????????Netron???????????????????????????????????????node
            let op = {a_netronNode: node};
            if (node._type === 'Conv' || node._type == 'Convolution') {
                // ????????????????????????????????????dsconv
                this.rky_CheckGroupConv(node, op);                
            } else {
                op._name = node._type.toUpperCase();
            }

            let isNewTns = true; // ????????????????????????tensor
            // rky: ???????????????????????????node????????????tensors
            for (let input of node._inputs) {
                for(let arg of input._arguments)
                {
                    isNewTns = true;
                    // rky: ????????????tensor???????????????????????????5
                    for (let t of sg_aryTensors) {
                        if (t._name == arg._name) {
                            isNewTns = false;
                            // rky: ????????????????????????tensor, ????????????op????????????????????????
                            t._toOps.push(op);
                            isNewTns = false;
                            break;
                        }
                    }
                    // rky: ?????????????????????tensor
                    if (isNewTns == true) {
                        let tns = {netronTns: arg, name: arg._name, ndx:tnsNdx, toOps: [op], fromOps:[]};
                        this.rky_ParseTensor(tns, arg, nodes, sg_aryTensors, true);
                        if (tns.isGraphInput) {
                            sg_aryInpTensors.push(tns);
                        }
                        tnsNdx++;
                    }
                }

                // (input._visible && input._arguments._length === 1 && input._arguments[0]._initializer != null)
            }

            for (let output of node._outputs) {
                for (let arg of output._arguments) {
                    // rky: ????????????tensor???????????????????????????
                    isNewTns = true;
                    for (let t of sg_aryTensors) {
                        if (t._name == arg._name) {
                            isNewTns = false;
                            // rky: ????????????????????????tensor, ????????????op????????????????????????
                            t._fromOps.push(op);
                            isNewTns = false;
                            break;
                        }
                    }
                    if (isNewTns == true) {
                        let tns = {netronTns: arg, name: arg._name, ndx:tnsNdx, fromOps: [op], toOps:[]};
                        this.rky_ParseTensor(tns, arg, nodes, sg_aryTensors, false);
                        if (tns.isGraphOutput) {
                            sg_aryOutTensors.push(tns);
                        }
                        tnsNdx++;
                    }
                }
            }
            sg_aryOps.push(op);
            if (sg_aryOpTypes.indexOf(op._name) < 0) {
                sg_aryOpTypes.push(op._name);
            }
            nodeNdx++;
            //console.log('parsed op,', op._name);

        }
        //console.log(view._nodes._length);

        //Work starts here!!!!!!!!!!!!!!!
        console.log("start to generate tflite file");

        let tfjson = {
            "version": 3,
            "operator_codes":[],
            "subgraphs":[{
                "tensors":[],
                "inputs":[],
                "outputs":[],
                "operators":[]
            }],
            "description":"anymodel to tflite",
            "buffers":[]
        };

        //?????????????????????????????????????????????)
        for(let??t??of??sg_aryOpTypes){
            if (t == "EXPAND"){
                tfjson.operator_codes.push({"builtin_code":"EXPAND_DIMS"})
            }
            else if (t == "CONV"){
                tfjson.operator_codes.push({"builtin_code":"conv_2d"})
            }
            else if (t == "MAXPOOL"){
                tfjson.operator_codes.push({"builtin_code":"MAX_POOL_2D"})
            }
            else if (t == "AveragePool"){
                tfjson.operator_codes.push({"builtin_code":"average_pool_2d"})
            }
            else if (t == "ArgMax"){
                tfjson.operator_codes.push({"builtin_code":"arg_max"})
            }
            else if (t == "ArgMin"){
                tfjson.operator_codes.push({"builtin_code":"arg_min"})
            }
            else if (t == "Concat"){
                tfjson.operator_codes.push({"builtin_code":"concatenation"})
            }
            else if (t == "TransposeConv"){
                tfjson.operator_codes.push({"builtin_code":"convolution_2d_transpode_bias"})
            }
            else if (t == "DepthToSpace"){
                tfjson.operator_codes.push({"builtin_code":"depth_to_space"})
            }
            else if (t == "DequantizeLinear"){
                tfjson.operator_codes.push({"builtin_code":"dequantize"})
            }
            else if (t == "Max"){
                tfjson.operator_codes.push({"builtin_code":"Max"})
            }
            else if (t == "Min"){
                tfjson.operator_codes.push({"builtin_code":"minimum"})
            }
            else if (t == "MaxUnpool"){
                tfjson.operator_codes.push({"builtin_code":"max_unpooling_2d"})
            }
            else if (t == "NonMaxSuppression"){
                tfjson.operator_codes.push({"builtin_code":"non_max_suppression"})
            }
            else if (t == "QuantizeLinear"){
                tfjson.operator_codes.push({"builtin_code":"quantize"})
            }
            else if (t == "ReverseSequence"){
                tfjson.operator_codes.push({"builtin_code":"reverse_sequence"})
            }
            else{
                tfjson.operator_codes.push({ "builtin_code" : t })
            }
        }

        
        //??????tensor
        let count =-1;
        for(let t of sg_aryTensors){
            count++;
            let nt={};
            nt.shape=[];
            if(t.data!=null&&t.data.shape!=null&&t.data.shape._shape!=null&&t.data.shape._shape.dim!=null){
                for (let ls of t.data.shape._shape.dim){
                        nt.shape.push(ls.size.low);
                }
            }
            else if(t.data!=null&&t.data.shape!=null&&t.data.shape._dimensions!=null){
                for (let ls of t.data.shape._dimensions){
                        nt.shape.push(ls.low);
                }
            }
            else if(t.netronTns!=null&&t.netronTns._type!=null&&t.netronTns._type._shape!=null&&t.netronTns._type._shape._dimensions!=null){
                for (let ls of t.netronTns._type._shape._dimensions){
                        nt.shape.push(ls.low);
                }
            }
            
            nt.name=t.netronTns._name;
            if(t.isGraphInput){
                tfjson.subgraphs[0].inputs.push(count);
            }
            if(t.isGraphOutput){
                tfjson.subgraphs[0].outputs.push(count);
            }
            nt.type=t.netronTns.dtype;
            if(t.netronTns._initializer!=null){
                if(t.netronTns._initializer._buffer!=null){
                    let lss=Object.values(t.netronTns._initializer._buffer);
                    let lsss={"data":lss};
                    tfjson.buffers.push(lsss);
                }
                else if(t.netronTns._initializer._values.buffer!=null){
                    let lss=Object.values(t.netronTns._initializer._values.buffer);
                    let lsss={"data":lss};
                    tfjson.buffers.push(lsss);
                }
                else {
                    let lss=[0];
                    let lsss={"data":lss};
                    tfjson.buffers.push(lsss);
                }
                
            }
            else {
                let lss=[];
                let lsss={"data":lss};
                tfjson.buffers.push(lsss);
            }
            nt.buffer=tfjson.buffers.length-1;

            // quantization
            // quantization(nt, t);

            tfjson.subgraphs[0].tensors.push(nt);
        }

        //?????????????????????tensor
        for(let i=0;i<sg_aryOps.length;i++){
            let op=sg_aryOps[i];
            let n_op={"inputs":[],"outputs":[]};
            for(let op_in of op.a_netronNode._inputs){
                if(op_in._arguments[0]!=null&&op_in._arguments[0]._name!=null){
                    n_op.inputs.push(check_tensor_id(op_in._arguments[0]._name));
                }
            }
            for(let op_out of op.a_netronNode._outputs){
                if(op_out._arguments[0]!=null&&op_out._arguments[0]._name!=null){
                    n_op.outputs.push(check_tensor_id(op_out._arguments[0]._name));
                }
            }
            tfjson.subgraphs[0].operators.push(n_op);
        }
        function check_tensor_id(tensor_name){
            for(let i =0;i< tfjson.subgraphs[0].tensors.length;i++){
                if(tensor_name==tfjson.subgraphs[0].tensors[i].name){
                    return i;
                }
            }
            return -1;
        }
        console.log("finish generating");


        //???????????????
        //tfjson.buffers=[];
        fs.writeFile('test.json', JSON.stringify(tfjson), err => {
            if (err) {
                console.error(err)
                return
            }
            console.error("??????????????????")//?????????????????????
        })
    }


    _parseCommandLine(argv) {
        let open = false;
        if (argv.length > 1) {
            for (const arg of argv.slice(1)) {
                if (!arg.startsWith('-')) {
                    const extension = arg.split('.').pop().toLowerCase();
                    if (extension != '' && extension != 'js' && fs.existsSync(arg) && fs.statSync(arg).isFile()) {
                        this._openFile(arg);
                        open = true;
                    }
                }
            }
        }
        return open;
    }

    _ready() {
        this._configuration.load();
        if (!this._configuration.has('userId')) {
            this._configuration.set('userId', this._uuid());
        }
        if (this._openFileQueue) {
            const queue = this._openFileQueue;
            this._openFileQueue = null;
            while (queue.length > 0) {
                const file = queue.shift();
                this._openFile(file);
            }
        }
        if (this._views.count == 0) {
            this._views.openView();
        }
        this._resetMenu();
        this._views.on('active-view-changed', () => {
            this._updateMenu();
        });
        this._views.on('active-view-updated', () => {
            this._updateMenu();
        });
    }

    _uuid() {
        const buffer = new Uint8Array(16);
        require("crypto").randomFillSync(buffer);
        buffer[6] = buffer[6] & 0x0f | 0x40;
        buffer[8] = buffer[8] & 0x3f | 0x80;
        const text = Array.from(buffer).map((value) => value < 0x10 ? '0' + value.toString(16) : value.toString(16)).join('');
        return text.slice(0, 8) + '-' + text.slice(8, 12) + '-' + text.slice(12, 16) + '-' + text.slice(16, 20) + '-' + text.slice(20, 32);
    }

    _openFileDialog() {
        const showOpenDialogOptions = {
            properties: [ 'openFile' ],
            filters: [
                { name: 'All Model Files (????????????)',  extensions: [
                    'onnx', 'pb',
                    'h5', 'hd5', 'hdf5', 'json', 'keras',
                    'mlmodel',
                    'caffemodel',
                    'model', 'dnn', 'cmf', 'mar', 'params',
                    'pdmodel', 'pdparams',
                    'meta',
                    'tflite', 'lite', 'tfl',
                    'armnn', 'mnn', 'nn', 'uff', 'uff.txt', 'rknn', 'xmodel',
                    'ncnn', 'param', 'tnnproto', 'tmfile', 'ms',
                    'pt', 'pth', 't7',
                    'pkl', 'joblib',
                    'pbtxt', 'prototxt',
                    'cfg', 'xml',
                    'zip', 'tar' ] }
            ]
        };
        const selectedFiles = electron.dialog.showOpenDialogSync(showOpenDialogOptions);
        if (selectedFiles) {
            for (const file of selectedFiles) {
                this._openFile(file);
            }
        }
    }

    _openFile(file) {
        if (this._openFileQueue) {
            this._openFileQueue.push(file);
            return;
        }
        if (file && file.length > 0 && fs.existsSync(file) && fs.statSync(file).isFile()) {
            // find existing view for this file
            let view = this._views.find(file);
            // find empty welcome window
            if (view == null) {
                view = this._views.find(null);
            }
            // create new window
            if (view == null) {
                view = this._views.openView();
            }
            this._loadFile(file, view);
        }
    }

    _loadFile(file, view) {
        const recents = this._configuration.get('recents').filter(recent => file != recent.path);
        view.open(file);
        recents.unshift({ path: file });
        if (recents.length > 9) {
            recents.splice(9);
        }
        this._configuration.set('recents', recents);
        this._resetMenu();
    }

    _dropFiles(sender, files) {
        let view = this._views.from(sender);
        for (const file of files) {
            if (view) {
                this._loadFile(file, view);
                view = null;
            }
            else {
                this._openFile(file);
            }
        }
    }
    _exportParam()
    {
        
    }
    _export() {
        const view = this._views.activeView;
        if (view && view.path) {
            let defaultPath = 'Untitled';
            const file = view.path;
            const lastIndex = file.lastIndexOf('.');
            if (lastIndex != -1) {
                defaultPath = file.substring(0, lastIndex);
            }
            const owner = electron.BrowserWindow.getFocusedWindow();
            const showSaveDialogOptions = {
                title: 'Export',
                defaultPath: defaultPath,
                buttonLabel: 'Export',
                filters: [
                    { name: 'PNG', extensions: [ 'png' ] },
                    { name: 'SVG', extensions: [ 'svg' ] }
                ]
            };
            const selectedFile = electron.dialog.showSaveDialogSync(owner, showSaveDialogOptions);
            if (selectedFile) {
                view.execute('export', { 'file': selectedFile });
            }
        }
    }

    service(name) {
        if (name == 'configuration') {
            return this._configuration;
        }
        return undefined;
    }

    execute(command, data) {
        const view = this._views.activeView;
        if (view) {
            view.execute(command, data || {});
        }
        this._updateMenu();
    }

    _reload() {
        const view = this._views.activeView;
        if (view && view.path) {
            this._loadFile(view.path, view);
        }
    }

    _checkForUpdates() {
        if (!electron.app.isPackaged) {
            return;
        }
        const autoUpdater = updater.autoUpdater;
        if (autoUpdater.app && autoUpdater.app.appUpdateConfigPath && !fs.existsSync(autoUpdater.app.appUpdateConfigPath)) {
            return;
        }
        const promise = autoUpdater.checkForUpdates();
        if (promise) {
            promise.catch((error) => {
                console.log(error.message);
            });
        }
    }

    get package() {
        if (!this._package) {
            const file = path.join(path.dirname(__dirname), 'package.json');
            const data = fs.readFileSync(file);
            this._package = JSON.parse(data);
            this._package.date = new Date(fs.statSync(file).mtime);
        }
        return this._package;
    }

    _about() {
        let dialog = null;
        const options = {
            show: false,
            backgroundColor: electron.nativeTheme.shouldUseDarkColors ? '#2d2d2d' : '#e6e6e6',
            width: 400,
            height: 250,
            center: true,
            minimizable: false,
            maximizable: false,
            useContentSize: true,
            resizable: true,
            fullscreenable: false,
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: true,
            }
        };
        if (process.platform === 'darwin') {
            options.title = '';
            dialog = Application._aboutDialog;
        }
        else {
            options.title = 'About ' + electron.app.name;
            options.parent = electron.BrowserWindow.getFocusedWindow();
            options.modal = true;
            options.showInTaskbar = false;
        }
        if (process.platform === 'win32') {
            options.type = 'toolbar';
        }
        if (!dialog) {
            dialog = new electron.BrowserWindow(options);
            if (process.platform === 'darwin') {
                Application._aboutDialog = dialog;
            }
            dialog.removeMenu();
            dialog.excludedFromShownWindowsMenu = true;
            dialog.webContents.on('new-window', (event, url) => {
                if (url.startsWith('http://') || url.startsWith('https://')) {
                    event.preventDefault();
                    electron.shell.openExternal(url);
                }
            });
            let content = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf-8');
            content = content.replace('{version}', this.package.version);
            content = content.replace('<title>Netron</title>', '');
            content = content.replace('<body class="welcome spinner">', '<body class="about desktop">');
            content = content.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
            content = content.replace(/<link.*>/gi, '');
            dialog.once('ready-to-show', () => {
                dialog.resizable = false;
                dialog.show();
            });
            dialog.on('close', function() {
                electron.globalShortcut.unregister('Escape');
                Application._aboutDialog = null;
            });
            dialog.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(content));
            electron.globalShortcut.register('Escape', function() {
                dialog.close();
            });
        }
        else {
            dialog.show();
        }
    }

    _updateMenu() {
        const window = electron.BrowserWindow.getFocusedWindow();
        this._menu.update({
            window: window,
            webContents: window ? window.webContents : null,
            view: this._views.activeView
        }, this._views.views.map((view) => view.window));
    }

    _resetMenu() {
        const menuRecentsTemplate = [];
        if (this._configuration.has('recents')) {
            let recents = this._configuration.get('recents');
            recents = recents.filter(recent => fs.existsSync(recent.path) && fs.statSync(recent.path).isFile());
            if (recents.length > 9) {
                recents.splice(9);
            }
            this._configuration.set('recents', recents);
            for (let i = 0; i < recents.length; i++) {
                const recent = recents[i];
                menuRecentsTemplate.push({
                    file: recent.path,
                    label: Application.minimizePath(recent.path),
                    accelerator: ((process.platform === 'darwin') ? 'Cmd+' : 'Ctrl+') + (i + 1).toString(),
                    click: (item) => { this._openFile(item.file); }
                });
            }
        }

        const menuTemplate = [];

        if (process.platform === 'darwin') {
            menuTemplate.unshift({
                label: electron.app.name,
                submenu: [
                    {
                        label: 'About ' + electron.app.name,
                        click: () => this._about()
                    },
                    { type: 'separator' },
                    { role: 'hide' },
                    { role: 'hideothers' },
                    { role: 'unhide' },
                    { type: 'separator' },
                    { role: 'quit' }
                ]
            });
        }

        menuTemplate.push({
            label: '&File',
            submenu: [
                {
                    label: '&Open...',
                    accelerator: 'CmdOrCtrl+O',
                    click: () => { this._openFileDialog(); }
                },
                {
                    label: 'Open &Recent',
                    submenu: menuRecentsTemplate
                },
                { type: 'separator' },
                {
                    id: 'file.export',
                    label: '&Export...',
                    accelerator: 'CmdOrCtrl+Shift+E',
                    click: () => this._export(),
                },
                { type: 'separator' },
                {
                    id: 'file.exportParam',
                    label: '&Export Param...',
                    accelerator: 'CmdOrCtrl+Shift+F',
                    enabled:false,
                    click: () => this.execute('export_param', null),
                },
                { type: 'separator' },
                { role: 'close' },
            ]
        });

        if (process.platform !== 'darwin') {
            menuTemplate.slice(-1)[0].submenu.push(
                { type: 'separator' },
                { role: 'quit' }
            );
        }

        if (process.platform == 'darwin') {
            electron.systemPreferences.setUserDefault('NSDisabledDictationMenuItem', 'boolean', true);
            electron.systemPreferences.setUserDefault('NSDisabledCharacterPaletteMenuItem', 'boolean', true);
        }

        menuTemplate.push({
            label: '&Edit',
            submenu: [
                {
                    id: 'edit.cut',
                    label: 'Cu&t',
                    accelerator: 'CmdOrCtrl+X',
                    click: () => this.execute('cut', null),
                },
                {
                    id: 'edit.copy',
                    label: '&Copy',
                    accelerator: 'CmdOrCtrl+C',
                    click: () => this.execute('copy', null),
                },
                {
                    id: 'edit.paste',
                    label: '&Paste',
                    accelerator: 'CmdOrCtrl+V',
                    click: () => this.execute('paste', null),
                },
                {
                    id: 'edit.select-all',
                    label: 'Select &All',
                    accelerator: 'CmdOrCtrl+A',
                    click: () => this.execute('selectall', null),
                },
                { type: 'separator' },
                {
                    id: 'edit.find',
                    label: '&Find...',
                    accelerator: 'CmdOrCtrl+F',
                    click: () => this.execute('find', null),
                }
            ]
        });

        const viewTemplate = {
            label: '&View',
            submenu: [
                {
                    id: 'view.show-attributes',
                    accelerator: 'CmdOrCtrl+D',
                    click: () => this.execute('toggle-attributes', null),
                },
                {
                    id: 'view.show-initializers',
                    accelerator: 'CmdOrCtrl+I',
                    click: () => this.execute('toggle-initializers', null),
                },
                {
                    id: 'view.show-names',
                    accelerator: 'CmdOrCtrl+U',
                    click: () => this.execute('toggle-names', null),
                },
                {
                    id: 'view.show-horizontal',
                    accelerator: 'CmdOrCtrl+K',
                    click: () => this.execute('toggle-direction', null),
                },
                { type: 'separator' },
                {
                    id: 'view.reload',
                    label: '&Reload',
                    accelerator: (process.platform === 'darwin') ? 'Cmd+R' : 'F5',
                    click: () => this._reload(),
                },
                { type: 'separator' },
                {
                    id: 'view.reset-zoom',
                    label: 'Actual &Size',
                    accelerator: 'Shift+Backspace',
                    click: () => this.execute('reset-zoom', null),
                },
                {
                    id: 'view.zoom-in',
                    label: 'Zoom &In',
                    accelerator: 'Shift+Up',
                    click: () => this.execute('zoom-in', null),
                },
                {
                    id: 'view.zoom-out',
                    label: 'Zoom &Out',
                    accelerator: 'Shift+Down',
                    click: () => this.execute('zoom-out', null),
                },
                { type: 'separator' },
                {
                    id: 'view.show-properties',
                    label: '&Properties...',
                    accelerator: 'CmdOrCtrl+Enter',
                    click: () => this.execute('show-properties', null),
                }
            ]
        };
        if (!electron.app.isPackaged) {
            viewTemplate.submenu.push({ type: 'separator' });
            viewTemplate.submenu.push({ role: 'toggledevtools' });
        }
        menuTemplate.push(viewTemplate);

        if (process.platform === 'darwin') {
            menuTemplate.push({
                role: 'window',
                submenu: [
                    { role: 'minimize' },
                    { role: 'zoom' },
                    { type: 'separator' },
                    { role: 'front'}
                ]
            });
        }

        const helpSubmenu = [
            {
                label: '&Search Feature Requests',
                click: () => { electron.shell.openExternal('https://www.github.com/' + this.package.repository + '/issues'); }
            },
            {
                label: 'Report &Issues',
                click: () => { electron.shell.openExternal('https://www.github.com/' + this.package.repository + '/issues/new'); }
            }
        ];

        if (process.platform != 'darwin') {
            helpSubmenu.push({ type: 'separator' });
            helpSubmenu.push({
                label: 'About ' + electron.app.name,
                click: () => this._about()
            });
        }

        menuTemplate.push({
            role: 'help',
            submenu: helpSubmenu
        });

        const commandTable = new Map();
        commandTable.set('file.export', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('file.exportParam', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });        
        commandTable.set('edit.cut', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('edit.copy', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('edit.paste', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('edit.select-all', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('edit.find', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('view.show-attributes', {
            enabled: (context) => { return context.view && context.view.path ? true : false; },
            label: (context) => { return !context.view || !context.view.get('show-attributes') ? 'Show &Attributes' : 'Hide &Attributes'; }
        });
        commandTable.set('view.show-initializers', {
            enabled: (context) => { return context.view && context.view.path ? true : false; },
            label: (context) => { return !context.view || !context.view.get('show-initializers') ? 'Show &Initializers' : 'Hide &Initializers'; }
        });
        commandTable.set('view.show-names', {
            enabled: (context) => { return context.view && context.view.path ? true : false; },
            label: (context) => { return !context.view || !context.view.get('show-names') ? 'Show &Names' : 'Hide &Names'; }
        });
        commandTable.set('view.show-horizontal', {
            enabled: (context) => { return context.view && context.view.path ? true : false; },
            label: (context) => { return !context.view || !context.view.get('show-horizontal') ? 'Show &Horizontal' : 'Show &Vertical'; }
        });
        commandTable.set('view.reload', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('view.reset-zoom', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('view.zoom-in', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('view.zoom-out', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('view.show-properties', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });

        this._menu.build(menuTemplate, commandTable, this._views.views.map((view) => view.window));
        this._updateMenu();
    }

    static minimizePath(file) {
        if (process.platform != 'win32') {
            const homeDir = os.homedir();
            if (file.startsWith(homeDir)) {
                return '~' + file.substring(homeDir.length);
            }
        }
        return file;
    }

}

class View {

    constructor(owner) {
        this._owner = owner;
        this._ready = false;
        this._path = null;
        this._properties = new Map();

        const size = electron.screen.getPrimaryDisplay().workAreaSize;
        const options = {
            show: false,
            title: electron.app.name,
            backgroundColor: electron.nativeTheme.shouldUseDarkColors ? '#1d1d1d' : '#e6e6e6',
            icon: electron.nativeImage.createFromPath(path.join(__dirname, 'icon.png')),
            minWidth: 600,
            minHeight: 400,
            width: size.width > 1024 ? 1024 : size.width,
            height: size.height > 768 ? 768 : size.height,
            webPreferences: {
                preload: path.join(__dirname, 'electron.js'),
                nodeIntegration: true,
                contextIsolation: true
            }
        };
        if (this._owner.count > 0 && View._position && View._position.length == 2) {
            options.x = View._position[0] + 30;
            options.y = View._position[1] + 30;
            if (options.x + options.width > size.width) {
                options.x = 0;
            }
            if (options.y + options.height > size.height) {
                options.y = 0;
            }
        }
        this._window = new electron.BrowserWindow(options);
        View._position = this._window.getPosition();
        this._updateCallback = (event, data) => {
            if (event.sender == this._window.webContents) {
                this.update(data.name, data.value);
                this._raise('updated');
            }
        };
        electron.ipcMain.on('update', this._updateCallback);
        this._window.on('closed', () => {
            electron.ipcMain.removeListener('update', this._updateCallback);
            this._owner.closeView(this);
        });
        this._window.on('focus', () => {
            this._raise('activated');
        });
        this._window.on('blur', () => {
            this._raise('deactivated');
        });
        this._window.webContents.on('did-finish-load', () => {
            this._didFinishLoad = true;
        });
        this._window.webContents.on('new-window', (event, url) => {
            if (url.startsWith('http://') || url.startsWith('https://')) {
                event.preventDefault();
                electron.shell.openExternal(url);
            }
        });
        this._window.once('ready-to-show', () => {
            this._window.show();
        });
        const location = url.format({ protocol: 'file:', slashes: true, pathname: path.join(__dirname, 'electron.html') });
        this._window.loadURL(location);
    }

    get window() {
        return this._window;
    }

    get path() {
        return this._path;
    }

    open(file) {
        this._openPath = file;
        if (this._didFinishLoad) {
            this._window.webContents.send('open', { file: file });
        }
        else {
            this._window.webContents.on('did-finish-load', () => {
                this._window.webContents.send('open', { file: file });
            });
            const location = url.format({ protocol: 'file:', slashes: true, pathname: path.join(__dirname, 'electron.html') });
            this._window.loadURL(location);
        }
    }

    restore() {
        if (this._window) {
            if (this._window.isMinimized()) {
                this._window.restore();
            }
            this._window.show();
        }
    }

    match(path) {
        if (this._openPath) {
            if (path === null) {
                return false;
            }
            if (path === this._openPath) {
                return true;
            }
        }
        return this._path == path;
    }

    execute(command, data) {
        if (this._window && this._window.webContents) {
            this._window.webContents.send(command, data);
        }
    }

    update(name, value) {
        if (name === 'path') {
            if (value) {
                this._path = value;
                const title = Application.minimizePath(this._path);
                this._window.setTitle(process.platform !== 'darwin' ? title + ' - ' + electron.app.name : title);
                this._window.focus();
            }
            this._openPath = null;
            return;
        }
        this._properties.set(name, value);
    }

    get(name) {
        return this._properties.get(name);
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }
}

class ViewCollection {

    constructor() {
        this._views = [];
    }

    get views() {
        return this._views;
    }

    get count() {
        return this._views.length;
    }

    item(index) {
        return this._views[index];
    }

    openView() {
        const view = new View(this);
        view.on('activated', (sender) => {
            this._activeView = sender;
            this._raise('active-view-changed', { activeView: this._activeView });
        });
        view.on('updated', () => {
            this._raise('active-view-updated', { activeView: this._activeView });
        });
        view.on('deactivated', () => {
            this._activeView = null;
            this._raise('active-view-changed', { activeView: this._activeView });
        });
        this._views.push(view);
        this._updateActiveView();
        return view;
    }

    closeView(view) {
        for (let i = this._views.length - 1; i >= 0; i--) {
            if (this._views[i] == view) {
                this._views.splice(i, 1);
            }
        }
        this._updateActiveView();
    }

    find(path) {
        return this._views.find(view => view.match(path));
    }

    from(contents) {
        return this._views.find(view => view && view.window && view.window.webContents && view.window.webContents == contents);
    }

    get activeView() {
        return this._activeView;
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }

    _updateActiveView() {
        const window = electron.BrowserWindow.getFocusedWindow();
        const view = this._views.find(view => view.window == window) || null;
        if (view != this._activeView) {
            this._activeView = view;
            this._raise('active-view-changed', { activeView: this._activeView });
        }
    }
}

class ConfigurationService {

    load() {
        this._data = { 'recents': [] };
        const dir = electron.app.getPath('userData');
        if (dir && dir.length > 0) {
            const file = path.join(dir, 'configuration.json');
            if (fs.existsSync(file)) {
                const data = fs.readFileSync(file);
                if (data) {
                    try {
                        this._data = JSON.parse(data);
                    }
                    catch (error) {
                        // continue regardless of error
                    }
                }
            }
        }
    }

    save() {
        if (this._data) {
            const data = JSON.stringify(this._data, null, 2);
            if (data) {
                const dir = electron.app.getPath('userData');
                if (dir && dir.length > 0) {
                    const file = path.join(dir, 'configuration.json');
                    fs.writeFileSync(file, data);
                }
            }
        }
    }

    has(name) {
        return this._data && Object.prototype.hasOwnProperty.call(this._data, name);
    }

    set(name, value) {
        this._data[name] = value;
    }

    get(name) {
        return this._data[name];
    }

}

class MenuService {

    build(menuTemplate, commandTable, windows) {
        this._menuTemplate = menuTemplate;
        this._commandTable = commandTable;
        this._itemTable = new Map();
        for (const menu of menuTemplate) {
            for (const item of menu.submenu) {
                if (item.id) {
                    if (!item.label) {
                        item.label = '';
                    }
                    this._itemTable.set(item.id, item);
                }
            }
        }
        this._rebuild(windows);
    }

    update(context, windows) {
        if (!this._menu && !this._commandTable) {
            return;
        }
        if (this._updateLabel(context)) {
            this._rebuild(windows);
        }
        this._updateEnabled(context);
    }

    _rebuild(windows) {
        this._menu = electron.Menu.buildFromTemplate(this._menuTemplate);
        if (process.platform === 'darwin') {
            electron.Menu.setApplicationMenu(this._menu);
        }
        else {
            for (const window of windows) {
                window.setMenu(this._menu);
            }
        }
    }

    _updateLabel(context) {
        let rebuild = false;
        for (const entry of this._commandTable.entries()) {
            const menuItem = this._menu.getMenuItemById(entry[0]);
            const command = entry[1];
            if (command && command.label) {
                const label = command.label(context);
                if (label != menuItem.label) {
                    if (this._itemTable.has(entry[0])) {
                        this._itemTable.get(entry[0]).label = label;
                        rebuild = true;
                    }
                }
            }
        }
        return rebuild;
    }

    _updateEnabled(context) {
        for (const entry of this._commandTable.entries()) {
            const menuItem = this._menu.getMenuItemById(entry[0]);
            if (menuItem) {
                const command = entry[1];
                if (command.enabled) {
                    menuItem.enabled = command.enabled(context);
                }
            }
        }
    }
}

global.application = new Application();
