import json
import os
import tempfile
import webbrowser
import numpy as np

HTML_TEMPLATE = '''<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Hasty OrthoSlicer</title>
    <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
    <style>
      body { font-family: sans-serif; margin: 8px; }
      .row { display:flex; gap:8px; }
      .col { display:flex; flex-direction:column; gap:8px; }
      .plot { width: 480px; height: 480px; }
      #controls { margin-top: 8px; }
    </style>
  </head>
  <body>
    <h3 id="title">Hasty OrthoSlicer</h3>
    <div class="row">
      <div class="col">
        <div id="sag" class="plot"></div>
        <input id="sag_slider" type="range" min="0" max="0" value="0" />
      </div>
      <div class="col">
        <div id="cor" class="plot"></div>
        <input id="cor_slider" type="range" min="0" max="0" value="0" />
      </div>
      <div class="col">
        <div id="axi" class="plot"></div>
        <input id="axi_slider" type="range" min="0" max="0" value="0" />
      </div>
    </div>
    <div id="controls">
      <label>Volume 1 index: <input id="vol1" type="number" value="0" min="0" step="1" /></label>
      <label>Volume 2 index: <input id="vol2" type="number" value="0" min="0" step="1" /></label>
      <button id="update_vol">Update Volume</button>
    </div>

    <script>
    // DATA variable set by Python embedding
    const DATA = JSON.parse(document.getElementById('DATA_JSON').textContent);
    document.getElementById('title').textContent = DATA.title || 'Hasty OrthoSlicer';

    // data shape: [nx, ny, nz, v3, v4]
    let arr = DATA.data;
    const nx = arr.length;
    const ny = arr[0].length;
    const nz = arr[0][0].length;
    const v3 = arr[0][0][0] && Array.isArray(arr[0][0][0]) ? arr[0][0][0].length : 1;
    const v4 = arr[0][0][0] && Array.isArray(arr[0][0][0]) && Array.isArray(arr[0][0][0][0]) ? arr[0][0][0][0].length : 1;

    function getVolume(v1, v2) {
      return {v1: v1|0, v2: v2|0};
    }

    function safeExtract(v, vol){
      if (!Array.isArray(v)) return v;
      if (vol.v1 < v.length){
        const x = v[vol.v1];
        if (Array.isArray(x)){
          if (vol.v2 < x.length) return x[vol.v2];
          return x[0];
        }
        return x;
      }
      return v[0];
    }

    function getSagSlice(ix, vol) {
      const mat = [];
      for (let j=0;j<ny;j++){
        const row = [];
        for (let k=0;k<nz;k++){
          row.push(safeExtract(arr[ix][j][k], vol));
        }
        mat.push(row);
      }
      return mat;
    }
    function getCorSlice(iy, vol) {
      const mat = [];
      for (let i=0;i<nx;i++){
        const row = [];
        for (let k=0;k<nz;k++){
          row.push(safeExtract(arr[i][iy][k], vol));
        }
        mat.push(row);
      }
      return mat;
    }
    function getAxiSlice(iz, vol) {
      const mat = [];
      for (let i=0;i<nx;i++){
        const row = [];
        for (let j=0;j<ny;j++){
          row.push(safeExtract(arr[i][j][iz], vol));
        }
        mat.push(row);
      }
      return mat;
    }

    function makeHeatmap(mat, divid) {
      const data = [{ z: mat, type: 'heatmap', colorscale: 'Greys', showscale:false }];
      const layout = {margin:{l:0,r:0,t:0,b:0}};
      Plotly.newPlot(divid, data, layout, {staticPlot:false});
    }

    const sag_slider = document.getElementById('sag_slider');
    const cor_slider = document.getElementById('cor_slider');
    const axi_slider = document.getElementById('axi_slider');
    sag_slider.max = nx-1; cor_slider.max = ny-1; axi_slider.max = nz-1;

    let curVol = getVolume(0,0);
    makeHeatmap(getSagSlice(0, curVol), 'sag');
    makeHeatmap(getCorSlice(0, curVol), 'cor');
    makeHeatmap(getAxiSlice(0, curVol), 'axi');

    sag_slider.addEventListener('input', ()=>{
      const mat = getSagSlice(parseInt(sag_slider.value), curVol);
      Plotly.react('sag', [{z:mat, type:'heatmap', colorscale:'Greys', showscale:false}], {margin:{l:0,r:0,t:0,b:0}});
    });
    cor_slider.addEventListener('input', ()=>{
      const mat = getCorSlice(parseInt(cor_slider.value), curVol);
      Plotly.react('cor', [{z:mat, type:'heatmap', colorscale:'Greys', showscale:false}], {margin:{l:0,r:0,t:0,b:0}});
    });
    axi_slider.addEventListener('input', ()=>{
      const mat = getAxiSlice(parseInt(axi_slider.value), curVol);
      Plotly.react('axi', [{z:mat, type:'heatmap', colorscale:'Greys', showscale:false}], {margin:{l:0,r:0,t:0,b:0}});
    });

    document.getElementById('update_vol').addEventListener('click', ()=>{
      const v1 = parseInt(document.getElementById('vol1').value)||0;
      const v2 = parseInt(document.getElementById('vol2').value)||0;
      curVol = getVolume(v1,v2);
      sag_slider.dispatchEvent(new Event('input'));
      cor_slider.dispatchEvent(new Event('input'));
      axi_slider.dispatchEvent(new Event('input'));
    });
    </script>
  </body>
</html>
'''

def _prepare_data_for_embed(npdata):
    arr = np.asanyarray(npdata)
    if arr.ndim == 3:
        arr = arr[..., np.newaxis, np.newaxis]
    elif arr.ndim == 4:
        arr = arr[..., np.newaxis]
    elif arr.ndim > 5:
        raise RuntimeError('Can\'t embed data with >5 dims')
    return arr.tolist()


class HastyOrthoSlicerWeb:
    def __init__(self, data, title=None, blocking=False):
        self.title = title or 'Hasty OrthoSlicer'
        self.blocking = bool(blocking)
        self.data = data

    def show(self):
        pydata = _prepare_data_for_embed(self.data)
        payload = dict(title=self.title, data=pydata)
        tmpdir = tempfile.mkdtemp(prefix='hasty_ortho_')
        html_path = os.path.join(tmpdir, 'ortho.html')
        with open(html_path, 'w') as f:
            f.write('<div id="DATA_JSON" style="display:none">')
            f.write(json.dumps(payload))
            f.write('</div>\n')
            f.write(HTML_TEMPLATE)

        webbrowser.open('file://' + html_path)

        if self.blocking:
            try:
                input('OrthoSlicer open in browser. Press Enter to continue...')
            except KeyboardInterrupt:
                pass


def image_nd(img, title=None, blocking=False):
    data = np.asanyarray(img)
    if data.ndim == 3:
        data = data[None, None, ...]
    if data.ndim == 4:
        data = data[None,...]
    slicer = HastyOrthoSlicerWeb(data, title=title, blocking=blocking)
    slicer.show()

if __name__ == '__main__':
    import numpy as np
    d = np.random.rand(64,64,64)
    image_nd(d)
