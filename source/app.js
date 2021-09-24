/* jshint esversion: 6 */

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
        // rocky: todo: 拆解chain，创建新的中间输入和输出tensor，以及新的nodes
        curOutputs = outputs;
        newNodes = [node];
        for (let i=0; i<node._chain._length; i++) {
        }
        return newNodes;
    }

    rky_IsGraphInput(tns, nodes) {
        if (tns.netronTns._initializer != null) {
            // 不把weights / bias 当作输入变量
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
            // 不把weights / bias 当作输出变量
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
        op._name = 'CONV_2D';    // 先默认是普通Conv2D，后面再修正
        for (let att of node._attributes) {
            if (att._name == 'group') {
                if (att._value != 1) {
                    // 分组数不是1._检查是否和输入通道数相同
                    let ic = 0; // ic = input channels
                    for (let i = 0; i< node._inputs._length; i++) {
                        // 遍历节点输入列表
                        for (let arg of node._inputs[i]._arguments) {
                            // 遍历各输入的参数
                            if (arg._initializer == null){
                                // 没有初始化的不是权重
                                continue;
                            }                        
                            if (arg._type != null) {
                                // 这里隐含假设输入数据使用了CHW数据序
                                ic = arg._type._shape._dimensions[0];
                                break;
                            }
                        }
                        if (ic != 0) {
                            break;
                        }
                    }

                    // rky: 潜在bug: 只有使用Int64时才需要._low
                    if (att._value._low === ic._low) {
                        // 输入通道数和组数相同，这是PyTorch/Caffe/onnx 派系模拟dsconv的方式
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
                // rky: 注意：来自不同输入格式的属性可能不同，只需先支持onnx, caffemodel
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

    rky_SaveTfliteJson(view) {
        let sFileName = view._model_folder + view._modelfile + '._json';
        let model = view._model;
        let dct = {
            version: 3,
        }
        let sg_aryTensors = []; // 计算图中的全体tensor
        let sg_aryInpTensors = [];  // 计算图中的输入tensor
        let sg_aryOutTensors = [];  // 计算图中的输出tensor
        let sg_aryOps = [];     // 计算图中的算子
        let sg_aryOpTypes = []; // 计算图中的算子类型, 用于tflite中的OperatorCode
        // 计算图的名称 (假设模型文件中只有一个计算图)。
        let sg_sName = view._model_name;
        let sFmt = view._format;
        let isCHW = false;  // 是否为CHW数据序。我们需要使用HWC数据序

        // onnx使用CHW数据序
        if (sFmt.indexOf('onnx') >= 0) {
            isCHW = true;
        }

        if (model._graphs.length != 1) {
            // tflite只支持单计算图
            console.log('sub graph must be 1!');
            return;
        }
        let nodeNdx = 0;
        let tnsNdx = 0;
        let nodes = model._graphs[0]._nodes; // rky: netron的node是operator
        for (let node of nodes) {
            // 遍历各node (运算符）。默认情况下，Netron已经按拓扑排序的顺序迭代各node
            let op = {a_netronNode: node};
            if (node._type === 'Conv' || node._type == 'Convolution') {
                // 判断是否是分组卷积实现的dsconv
                this.rky_CheckGroupConv(node, op);                
            } else {
                op._name = node._type.toUpperCase();
            }

            let isNewTns = true; // 先假设是新出现的tensor
            // rky: 遍历所有被当作这个node的输入的tensors
            for (let input of node._inputs) {
                for(let arg of input._arguments)
                {
                    isNewTns = true;
                    // rky: 检查这个tensor是不是已经出现过了5
                    for (let t of sg_aryTensors) {
                        if (t._name == arg._name) {
                            isNewTns = false;
                            // rky: 这是已经出现过的tensor, 在另一个op上又被当作输入了
                            t._toOps.push(op);
                            isNewTns = false;
                            break;
                        }
                    }
                    // rky: 找到了一个新的tensor
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
                    // rky: 检查这个tensor是不是已经出现过了
                    isNewTns = true;
                    for (let t of sg_aryTensors) {
                        if (t._name == arg._name) {
                            isNewTns = false;
                            // rky: 这是已经出现过的tensor, 在另一个op上又被当作输入了
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
            console.log('parsed op,', op._name);
        }
        console.log("111111");
        //console.log(view._nodes._length);
        let tfjson = {"version": 3,
            "operator_codes":[],
            "subgraphs":[{"tensors":[],"inputs":[],"outputs":[],"operators":[]}],
            "description":"a",
            "buffers":[]
        };
        for(let t of sg_aryOpTypes){
            tfjson.operator_codes.push({"builtin_code":t});
        }
        console.log(tfjson);
        for(let t of sg_aryTensors){
            // TODO: 此处的判断有问题
            if(t.netronTns._type == null){
                tfjson.subgraphs[0].tensors.push({
                    "shape":[0],
                    "buffer":1,
                    "type" : t.toOps[0].a_netronNode._attributes[0]._value,
                    "name":t.netronTns._name,
                    "quantization":{}
                });
            }
            else{
                let s ;
                for(let i=0;i<t.netronTns._type._shape._shape.dim.length;i++){
                    if(t.netronTns._type._shape._shape.dim[i].size.low!=0){
                        s=s+t.netronTns._type._shape._shape.dim[i].size.low+",";
                    }
                    if(t.netronTns._type._shape._shape.dim[i].size.high!=0){
                        s=s+t.netronTns._type._shape._shape.dim[i].size.high+",";
                    }
                }
                console.log(s);
                tfjson.subgraphs[0].tensors.push({
                    "shape": [s],
                    "buffer": 1,
                    "type" : t.toOps[0].a_netronNode._attributes[0]._value,
                    "name": t.netronTns._name,
                    "quantization":{
                    }
                });
            }
        }
        console.log(tfjson);
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
                { name: 'All Model Files (咱啥都认)',  extensions: [
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
