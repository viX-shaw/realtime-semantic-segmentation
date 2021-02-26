import React from "react";
import ReactDOM from "react-dom";
import * as tf from '@tensorflow/tfjs';
//uncomment below 2 lines for WASM Backend
// import '@tensorflow/tfjs-backend-wasm';
// import {setWasmPaths} from '@tensorflow/tfjs-backend-wasm';
import "./styles.css";

tf.enableProdMode();
tf.setBackend('webgl');

const pascalvoc = [[ 0,0,0 ],[ 128,0,0 ],[ 0,128,0 ],
                    [ 128,128,0 ],[ 0,0,128 ],[ 128,0,128 ],
                    [ 0,128,128 ],[ 128,128,128 ],[ 64,0,0 ],
                    [ 192,0,0 ],[ 64,128,0 ],[ 192,128,0 ],
                    [ 64,0,128 ],[ 192,0,128 ],[ 64,128,128 ],
                    [ 192,128,128 ],[ 0,64,0 ],[ 128,64,0 ],
                    [ 0,192,0 ],[ 128,192,0 ],[ 0,64,128 ],
                    [ 128,64,128 ],[ 0,192,128 ],[ 128,192,128 ],
                    [ 64,64,0 ],[ 192,64,0 ],[ 64,192,0 ],
                    [ 192,192,0 ],[ 64,64,128 ],[ 192,64,128 ],
                    [ 64,192,128 ],[ 192,192,128 ],[ 0,0,64 ],
                    [ 128,0,64 ],[ 0,128,64 ],[ 128,128,64 ],
                    [ 0,0,192 ],[ 128,0,192 ],[ 0,128,192 ],
                    [ 128,128,192 ],[ 64,0,64 ]];


async function load_model() {
  //uncomment below 2 lines for WASM Backend
  // setWasmPaths('http://localhost:9001/'); // or tf.wasm.setWasmPaths when using <script> tags.
  // await tf.setBackend('wasm');
  const model = await tf.loadLayersModel("http://localhost:9001/model.json");
  return model;
}

const modelPromise = load_model();

class App extends React.Component {
  videoRef = React.createRef();
  canvasRef = React.createRef();
  predictions = null
  prevImg = null
  
  componentDidMount() {

    var loop_count = 0

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "user"
          }
        })
        .then(stream => {
          window.stream = stream;
          this.videoRef.current.srcObject = stream;
          return new Promise((resolve, reject) => {
            this.videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          });
        });
      Promise.all([modelPromise, webCamPromise])
        .then(values => {
          this.detectFrame(this.videoRef.current, values[0], loop_count);
        })
        .catch(error => {
          console.error(error);
        });
    }
  }

  detectFrame = (video, model, loop_count) => {
    tf.engine().startScope();
    let infVal = null;
    let image = tf.image.resizeBilinear(tf.browser.fromPixels(video).toFloat(), [480, 480]);
    let img = tf.image.resizeBilinear(image, [160, 160])
    // console.log(img.shape)
    if (this.prevImg !== null) {
      // get the diff of current image and prev image
      // compute a value which will determine whether to perform inference in current iteration
      infVal = img.sub(tf.tensor(this.prevImg)).mean().abs().dataSync()[0]
    }
    if (loop_count === 0 || infVal > 0.25) {//loop_count % 3 !== 0) {
      // console.log("pred", loop_count)
      this.predictions = model.predict(this.process_input(img)).arraySync();
    }
    this.renderPredictions(image, tf.tensor(this.predictions));
    loop_count = loop_count + 1
    if (loop_count > 10000) {
      loop_count = 0
    }
    this.prevImg = img.arraySync()
    requestAnimationFrame(() => {
      // this.detectFrame(video, model, loop_count);
      setTimeout(this.detectFrame, 85, video, model, loop_count)
    });
    tf.engine().endScope();
  };

  process_input(img){
    // const img = tf.browser.fromPixels(video_frame).toFloat();
    const scale = tf.scalar(255.);
    const mean = tf.tensor3d([0.485, 0.456, 0.406], [1,1,3]);
    const std = tf.tensor3d([0.229, 0.224, 0.225], [1,1,3]);
    const normalised = img.div(scale).sub(mean).div(std);
    const batched = normalised.transpose([2,0,1]).expandDims();
    return batched;
  };
  
  renderPredictions = async (img, predictions) => {
    let dim = 480
    const img_shape = [dim, dim]
    const offset = 0;
    const segmPred = tf.image.resizeBilinear(predictions.transpose([0,2,3,1]),
                                              img_shape);
    let back_img_pixels = tf.browser.fromPixels(document.getElementById("myimg")).toFloat();
    // img = tf.image.resizeBilinear(img, img_shape)
    let segmMask = segmPred.argMax(3).reshape(img_shape);
      //Class person id - 0
    let personMask = tf.fill([dim,dim], 0, 'int32')
    segmMask = segmMask.equal(personMask)

    //Change Background    
    segmMask = segmMask.broadcastTo([3, dim, dim]).transpose([1,2,0])
    let final_img = back_img_pixels.where(segmMask, img)
    let alphaChannel = tf.fill([dim,dim,1], 255, 'int32') 
    final_img = tf.concat([final_img, alphaChannel], 2)
    // final_img = tf.image.resizeBilinear(final_img, [480, 480]);
    let img_buff = await final_img.data()

    //Background blur
    // segmMask = segmMask.broadcastTo([1, 480, 480]).transpose([1,2,0])
    // // let alphaChannel = tf.fill([480,480,1], 190).where(segmMask, tf.fill([480,480,1], 255)) 
    // let alphaChannel = tf.randomUniform([480, 480, 1], 2.5, 5, 'float32', 42).exp().ceil().where(segmMask, tf.fill([480,480,1], 255)) 
    // let final_img = tf.concat([img, alphaChannel], 2)
    // let img_buff = await final_img.data()
    
    const bytes = Uint8ClampedArray.from(img_buff)

    // const width = segmMask.shape.slice(0, 1);
    // const height = segmMask.shape.slice(1, 2);
    // const data = await segmMask.data();
    // const bytes = new Uint8ClampedArray(width * height * 4);
    // for (let i = 0; i < height * width; ++i) {
    //   const partId = data[i];
    //   const j = i * 4;
    //   if (partId === -1) {
    //       bytes[j + 0] = 255;
    //       bytes[j + 1] = 255;
    //       bytes[j + 2] = 255;
    //       bytes[j + 3] = 255;
    //   } else {
    //       const color = pascalvoc[partId + offset];

    //       if (!color) {
    //           throw new Error(`No color could be found for part id ${partId}`);
    //       }
    //       bytes[j + 0] = color[0];
    //       bytes[j + 1] = color[1];
    //       bytes[j + 2] = color[2];
    //       bytes[j + 3] = 255;
    //   }
    // }
    const out = new ImageData(bytes, 480, 480);
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.scale(1.5, 1.5);
    ctx.putImageData(out, 520, 60);
  };

  render() {
    return (
      <div>
        <h1>Real-Time Semantic Segmentation</h1>
        <h3>Refine Net</h3>
        {/* <img id="myimg" src={process.env.PUBLIC_URL + '/images/sky.jpg'} 
         */}
        <img id="myimg" src={process.env.PUBLIC_URL + '/images/walls.jpeg'} 
          style={{opacity: 0}} alt="sky" height="480" width="480"/>
        <video
          style={{marginTop:480, height: '600px', width: "480px", opacity: 0}}
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          width= "480"
          height= "480"
        />
        <canvas
          className="size"
          ref={this.canvasRef}
          width="960"
          height="480"
        />
      </div>
    );
  }

}
const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
