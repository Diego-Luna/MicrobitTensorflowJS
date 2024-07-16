/* eslint-disable no-alert */
/* eslint-disable no-undef */
/* eslint-disable func-style */
/* eslint-disable require-jsdoc */
const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const formatMessage = require('format-message');
const tf = require('@tensorflow/tfjs');

let port;
let writer;
let reader;
let isConnected = false;
const trainingData = [];
let model;
let classNames = [];
let numClasses = 0;

async function connect () {
    if (!('serial' in navigator)) {
        alert('Tu navegador no soporta la API Web Serial');
        return;
    }

    try {
        port = await navigator.serial.requestPort();
        await port.open({baudRate: 115200});

        const textEncoder = new TextEncoderStream();
        const writableStreamClosed = textEncoder.readable.pipeTo(port.writable);
        writer = textEncoder.writable.getWriter();

        const textDecoder = new TextDecoderStream();
        const readableStreamClosed = port.readable.pipeTo(textDecoder.writable);
        reader = textDecoder.readable.getReader();

        isConnected = true;
        console.log('Conectado a la micro:bit');
    } catch (error) {
        console.error('Error al conectar:', error);
    }
}

async function disconnect () {
    if (isConnected && port) {
        try {
            await writer.close();
            await reader.cancel();
            await port.close();
            isConnected = false;
            console.log('Conexión cerrada');
        } catch (error) {
            console.error('Error al desconectar:', error);
        }
    }
}

async function sendData (dataToSend) {
    if (isConnected && writer) {
        try {
            await writer.write(`${dataToSend}\n`);
            console.log('Datos enviados:', dataToSend);
        } catch (error) {
            console.error('Error al enviar datos:', error);
        }
    } else {
        console.log('No hay conexión activa');
    }
}

async function receiveData () {
    if (isConnected && reader) {
        try {
            const {value, done} = await reader.read();
            if (done) {
                console.log('Stream cerrado');
                return null;
            }
            console.log('Datos recibidos:', value);
            return value;
        } catch (error) {
            console.error('Error al recibir datos:', error);
            return null;
        }
    }
    return null;
}

async function collectTrainingData(classIndex, seconds) {
    const data = [];
    const start = Date.now();
    while ((Date.now() - start) < seconds * 1000) {
        const value = await receiveData();
        if (value !== null) {
            data.push({ input: parseFloat(value), label: classIndex });
        }
    }
    trainingData.push(...data);
    console.log("Datos recolectados para la clase", classIndex, ":", data);
}

function createModel (numClasses) {
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 16, activation: 'relu', inputShape: [1]}));
    model.add(tf.layers.dense({units: numClasses, activation: 'softmax'}));
    model.compile({optimizer: 'adam', loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy']});
    console.log("> createModel:", model);
    return model;
}

async function trainModel() {
    const xs = tf.tensor2d(trainingData.map(item => [item.input]), [trainingData.length, 1], 'float32');
    const ys = tf.tensor1d(trainingData.map(item => item.label), 'int32');
    await model.fit(xs, ys, { epochs: 50 });
    console.log("> trainModel");
}

async function predict(input) {
    const prediction = model.predict(tf.tensor2d([parseFloat(input)], [1, 1], 'float32'));
    console.log("> predict:", prediction.argMax(-1).dataSync()[0]);
    return prediction.argMax(-1).dataSync()[0];
}

const imgBLOC = 'data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iTGF5ZXJfMSIgZGF0YS1uYW1lPSJMYXllciAxIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDgwIDgwIj4KICA8ZGVmcz4KICAgIDxzdHlsZT4KICAgICAgLmNscy0xIHsKICAgICAgICBmaWxsOiAjZmZmOwogICAgICAgIHN0cm9rZS13aWR0aDogMHB4OwogICAgICB9CiAgICA8L3N0eWxlPgogIDwvZGVmcz4KICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xOC4yNywzNS45NGMtLjI3LS40MS0uNTQtLjgxLS44MS0xLjIxLS45Ni0xLjQ0LTEuOTItMi44OC0yLjg4LTQuMzItLjA3LS4xMS0uMTEtLjI5LS4zLS4yNS0uMjEuMDUtLjE1LjI0LS4xNS4zOCwwLDIuNjQsMCw1LjI4LDAsNy45MiwwLC4zMy0uMDkuNDItLjQxLjQxLTEuMDEtLjAyLTIuMDEtLjAyLTMuMDIsMC0uMjYsMC0uMzMtLjA2LS4zMy0uMzMsMC00LjgzLDAtOS42NiwwLTE0LjUsMC0uMjguMDktLjM1LjM2LS4zNSwxLjE5LjAxLDIuMzguMDEsMy41NywwLC4yNCwwLC4zNy4wOC40OS4yOSwxLjAyLDEuNzQsMi4wNiwzLjQ4LDMuMDksNS4yMi4zMy41Ni41MS41Ny44NiwwLDEuMDktMS43MiwyLjE4LTMuNDQsMy4yNS01LjE3LjE2LS4yNi4zNC0uMzUuNjQtLjM0LDEuMTMuMDIsMi4yNi4wMSwzLjQsMCwuMjcsMCwuMzkuMDQuMzkuMzYsMCw0LjgyLDAsOS42NCwwLDE0LjQ1LDAsLjMxLS4xMS4zNi0uMzkuMzYtMS4wNi0uMDEtMi4xMy0uMDItMy4xOSwwLS4zMiwwLS40LS4wOC0uMzktLjM5LjAxLTIuNjUsMC01LjMxLDAtNy45NiwwLS4xLDAtLjIsMC0uMjksMC0uMTMuMDUtLjMyLS4xMS0uMzctLjItLjA3LS4yNS4xNC0uMzMuMjYtMS4xNiwxLjgzLTIuMzEsMy42NS0zLjQ3LDUuNDgtLjA3LjExLS4xMS4yNS0uMjguMzRaIi8+CiAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNOS41Myw1My4wNWMtLjQ2LS42OS0uOS0xLjM0LTEuMzMtMS45OS0uNzctMS4xNS0xLjUzLTIuMy0yLjMtMy40NC0uMS0uMTUtLjE1LS40NC0uMzktLjM3LS4yNi4wOC0uMTQuMzYtLjE0LjU1LDAsMi41NC0uMDIsNS4wOCwwLDcuNjIsMCwuNDQtLjEzLjUzLS41NC41Mi0uOTUtLjAzLTEuOS0uMDItMi44NSwwLS4yNCwwLS4zNy0uMDItLjM3LS4zMy4wMS00LjgzLDAtOS42NiwwLTE0LjQ5LDAtLjI1LjA1LS4zNC4zMi0uMzMsMS4yLjAyLDIuNC4wMSwzLjYxLDAsLjI1LDAsLjM3LjEuNDkuMywxLjAxLDEuNzIsMi4wMywzLjQzLDMuMDUsNS4xNS4zOC42My41My42NC45MS4wMywxLjA5LTEuNzIsMi4xNy0zLjQ0LDMuMjUtNS4xNy4xMy0uMjEuMjctLjMuNTMtLjMsMS4xNi4wMSwyLjMyLjAxLDMuNDgsMCwuMjcsMCwuMzkuMDQuMzkuMzYtLjAxLDQuODIsMCw5LjYzLDAsMTQuNDUsMCwuMjktLjEuMzUtLjM2LjM1LTEuMDUtLjAxLTIuMS0uMDItMy4xNCwwLS4zNCwwLS40NC0uMDctLjQ0LS40My4wMi0yLjY3LDAtNS4zMywwLTgsMC0uMDgsMC0uMTcsMC0uMjUsMC0uMTMuMDQtLjMtLjE1LS4zNC0uMTctLjA0LS4yMS4xMS0uMjcuMjEtLjMxLjQ5LS42My45OS0uOTQsMS40OS0uOTIsMS40Ni0xLjg1LDIuOTItMi44MSw0LjQ0WiIvPgogIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTQ1LjU1LDQ0LjA4Yy0uNzQuODctMS40MSwxLjY3LTIuMDksMi40Ni0uNS41OS0uOTgsMS4xOC0xLjUsMS43NS0uMjguMzEtLjMuNTYtLjA5LjkzLDEuMTksMS45OSwyLjM1LDMuOTksMy41Miw1Ljk4LjA3LjEzLjE5LjI0LjE4LjQ0LTEuMywwLTIuNTksMC0zLjg5LDAtLjIsMC0uMjUtLjE1LS4zMi0uMjgtLjY1LTEuMDctMS4zLTIuMTQtMS45Ni0zLjIyLS4zMS0uNTEtLjQ1LS40OS0uODgtLjA4LS41OS41Ny0uODcsMS4yMi0uNzYsMi4wNS4wNS40MSwwLC44NC4wMSwxLjI2LDAsLjE5LS4wNS4yNy0uMjYuMjctMS4xMiwwLTIuMjQtLjAxLTMuMzUsMC0uMjYsMC0uMjUtLjEzLS4yNS0uMzEsMC0uOTEsMC0xLjgxLDAtMi43MiwwLTQuMDksMC04LjE4LDAtMTIuMjcsMC0uMzYuMDctLjQ4LjQ1LS40Ni45OS4wMywxLjk4LjAyLDIuOTgsMCwuMzQsMCwuNDUuMDcuNDUuNDMtLjAyLDIuMzYsMCw0LjcyLDAsNy4wOCwwLC4wOCwwLC4xNywwLC4yNSwwLC4xNC0uMDMuMy4xMy4zNi4xOS4wOC4yOC0uMDguMzctLjE5LjktMS4xNCwxLjgtMi4yOCwyLjY5LTMuNDIuMTctLjIyLjMzLS4zMi42Mi0uMzIsMS4yOC4wMiwyLjU3LDAsMy45NiwwWiIvPgogIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTU4LjI1LDMyLjhjMC0xLjc5LDAtMy41NywwLTUuMzYsMC0uMjkuMDQtLjQxLjM4LS40LDEuMDEuMDMsMi4wMS4wMiwzLjAyLDAsLjI4LDAsLjM3LjA4LjM1LjM1LS4wMi4yOSwwLC41OSwwLC44OCwwLC4xNi4wNS4zMS4yMi4zNy4xOC4wNy4zMy4wMS40NC0uMTQuMDQtLjA2LjA5LS4xMS4xMy0uMTYsMS40Mi0xLjc0LDQuMTktMS45NCw1Ljg1LS40Mi42OC42MiwxLjAzLDEuMzksMS4wNSwyLjMzLjA3LDIuNjUtLjA0LDUuMywwLDcuOTUsMCwuMzQtLjEuNDItLjQyLjQxLTEuMDUtLjAyLTIuMS0uMDEtMy4xNCwwLS4yNSwwLS4zNC0uMDYtLjM0LS4zMi4wMS0yLjA0LDAtNC4wOC4wMS02LjExLjAxLTEuNDgtMS4zNS0yLjQxLTIuNzItMS44Mi0uNzEuMzEtMS4wMS44Ny0xLjAxLDEuNjMsMCwxLjk4LDAsMy45NiwwLDUuOTVxMCwuNjctLjY5LjY3Yy0uOTIsMC0xLjg1LS4wMS0yLjc3LDAtLjMsMC0uMzctLjA5LS4zNy0uMzcuMDEtMS44MSwwLTMuNjMsMC01LjQ0WiIvPgogIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTM1LjcxLDM4Ljk4Yy0zLjY3LjE1LTYuOTQtMi4yOC02Ljk2LTUuOTktLjAxLTIuNTYsMS4zMS00LjM5LDMuNjEtNS40NywyLjY2LTEuMjUsNS44NS0uNTYsNy43MywxLjU5LDIuNzgsMy4xOSwxLjQsOC4wNi0yLjY4LDkuNS0uNjUuMjMtMS4zMi4zNi0xLjcuMzdaTTM1LjAyLDMwLjA3Yy0xLjQ3LDAtMi42NiwxLjI3LTIuNjUsMi44MywwLDEuNTYsMS4yMSwyLjgzLDIuNjcsMi44MiwxLjQ1LDAsMi42Ny0xLjMsMi42Ni0yLjg0LS4wMS0xLjU2LTEuMjEtMi44Mi0yLjY3LTIuODFaIi8+CiAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNNDMuMzcsMzMuMjZjLjA5LTIuMjcsMS4xNS0zLjk5LDMuMTItNS4xNCwyLjUxLTEuNDcsNS44Ny0xLjA0LDcuOS45OSwyLjYxLDIuNiwyLjM0LDYuNjUtLjYsOC45LTMuNDQsMi42My04LjYyLDEuMjEtMTAuMDctMi43Ni0uMTMtLjM2LS4yMi0uNzItLjI3LTEuMDktLjA0LS4yOS0uMDUtLjU4LS4wNy0uOVpNNTIuMzEsMzMuMjZjMC0xLjU3LTEuMTYtMi44NC0yLjYyLTIuODYtMS40Ni0uMDEtMi42NywxLjI2LTIuNjcsMi44MSwwLDEuNTgsMS4xNiwyLjg0LDIuNjMsMi44NCwxLjQ5LDAsMi42Ny0xLjIzLDIuNjctMi44WiIvPgogIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTMxLjExLDUxLjg2YzAsMS4yLS4wMSwyLjQsMCwzLjYsMCwuMzgtLjExLjUtLjUuNDktLjk5LS4wMy0xLjk4LS4wMi0yLjk4LDAtLjI2LDAtLjM3LS4wNC0uMzYtLjM0LDAtLjI2LjE2LS42NS0uMTEtLjc4LS4zMS0uMTUtLjQ4LjI2LS42OC40My0yLDEuNjktNC40OCwxLjA3LTUuNjctLjA5LTEuMDctMS4wNC0xLjE2LTIuMzQtLjgxLTMuNjkuMzQtMS4zMiwxLjMxLTIuMDQsMi42LTIuMjUsMS40OC0uMjUsMi45OC0uMjgsNC40Mi4yNi4zMi4xMi4zNi0uMDQuMzctLjI5LjAyLTEuMjQtLjg2LTIuMS0yLjI3LTIuMTctMS4wNy0uMDUtMi4xMS4xNC0zLjExLjU2LS4xNy4wNy0uMzEuMTctLjQtLjEyLS4yMy0uNzMtLjQ5LTEuNDUtLjc1LTIuMTctLjA2LS4xNy0uMDctLjI5LjEzLS4zNiwyLjI4LS43Nyw0LjYtMS4yMSw2Ljk4LS40OSwxLjI2LjM4LDIuMzIsMS4wMywyLjg2LDIuMzEuMjMuNTMuMjYsMS4wOC4yNywxLjY0LjAxLDEuMTYsMCwyLjMyLDAsMy40OFpNMjUuNTEsNTAuODJjLS42NS0uMDQtMS4zMy4xNi0xLjgyLjgxLS41OS43Ny0uMjUsMS44Ni42NywyLjE5LDEuNTguNTUsMy4xNy0uNjksMy4wMS0yLjM1LS4wMi0uMjQtLjEyLS4zOC0uMzYtLjQ0LS40NS0uMTItLjktLjE5LTEuNS0uMloiLz4KICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik01My43OSw1MS4wN2MtMS4xNiwwLTIuMzIsMC0zLjQ4LDAtLjUyLDAtLjY1LjItLjQ1LjY5LjU0LDEuMzIsMi4xNSwxLjk1LDMuNiwxLjQxLjUyLS4xOS45OS0uNDcsMS4zNi0uODcuMTgtLjE5LjI4LS4xOC40Ny0uMDEuNS40NCwxLjAxLjg5LDEuNTUsMS4yOC4zNC4yNS4yNS40MS4wMy42NC0uNzMuNzctMS42MiwxLjI3LTIuNjQsMS41NC0xLjk4LjUzLTMuOTIuNDktNS42OS0uNjUtMS44Mi0xLjE2LTIuNjMtMi44OC0yLjU0LTUuMDEuMTEtMi41LDEuMDctNC41NCwzLjQ2LTUuNjIsMS42LS43MywzLjI3LS42OSw0LjktLjEsMS44LjY1LDIuNjUsMi4xNCwzLjEzLDMuODcuMjMuODEuMjIsMS42NS4yMywyLjQ5LDAsLjI5LS4xLjM1LS4zNy4zNS0xLjE5LS4wMS0yLjM4LDAtMy41NiwwWk01MS44OCw0OS4xOGMuNjQsMCwxLjI4LDAsMS45MiwwLC4yLDAsLjMtLjA0LjI5LS4yNy0uMDUtMS4zLTEuMDctMi4yNC0yLjQ0LTIuMjMtMS4xNiwwLTEuODkuODMtMS45NSwyLjE1LS4wMS4yNy4wNi4zNi4zNC4zNS42MS0uMDIsMS4yMywwLDEuODQsMFoiLz4KICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik03My42NCw1NS45NmMtMS43OC0uMDQtMy40Mi0uNDItNC44Ny0xLjQyLS4yNy0uMTktLjMyLS4zNC0uMTMtLjYzLjM0LS41Mi42NS0xLjA3Ljk0LTEuNjMuMTUtLjI5LjI3LS4yOS41My0uMTIuODMuNTIsMS42OS45OSwyLjY3LDEuMTUuNTQuMDksMS4wOC4xMiwxLjYyLS4wMi4zNS0uMDkuNTYtLjMxLjYtLjY3LjA0LS4zNy0uMTctLjYyLS41LS43My0uODctLjMtMS43NS0uNTgtMi42Mi0uODgtLjM3LS4xMy0uNzMtLjI4LTEuMDktLjQzLTEuMjQtLjUzLTEuODgtMS40OC0xLjk0LTIuODMtLjA1LTEuMjcuNDMtMi4yOSwxLjQ3LTMuMDEuOTMtLjY0LDEuOTgtLjg3LDMuMDktLjksMS41Mi0uMDMsMi45Ny4yMSw0LjMyLjk3LjMzLjE5LjQ3LjM1LjIyLjczLS4zMi41LS42LDEuMDQtLjg2LDEuNTgtLjE2LjMzLS4yOS40LS42NC4yLS45LS41MS0xLjg1LS45Mi0yLjkzLS44NC0uMjQuMDItLjQ3LjA0LS43LjEyLS41My4xOS0uNjUuNzMtLjI2LDEuMTMuMjUuMjYuNTcuMzkuOS41MS43MS4yNiwxLjQ2LjM3LDIuMTkuNTcuNDMuMTIuODUuMjYsMS4yNC40OS45Ny41NSwxLjQ2LDEuMzgsMS40OCwyLjUuMDEuNSwwLDEtLjEyLDEuNS0uMzYsMS41NS0xLjQ4LDIuMjEtMi45MiwyLjQ5LS41OC4xMS0xLjE2LjEzLTEuNy4xNloiLz4KICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik01OS43Nyw1MC4yYzAtMS44LDAtMy42LDAtNS40LDAtLjMuMDYtLjQxLjM5LS40LDEuMDMuMDIsMi4wNy4wMiwzLjEsMCwuMjUsMCwuMzQuMDYuMzMuMzItLjAyLjU3LDAsMS4xNCwwLDEuNzIsMCwuMTItLjAzLjI3LjE0LjMuMTUuMDMuMTgtLjExLjIzLS4yLjIzLS40Mi41LS44MS43Ny0xLjE5LjYtLjgzLDEuNDYtMS4wNiwyLjQxLTEuMTMuMTktLjAxLjI1LjA1LjI0LjIzLDAsMS4yLDAsMi40LDAsMy42LDAsLjItLjA5LjIyLS4yNS4yMy0uMzkuMDMtLjc4LjA3LTEuMTUuMTktMS40LjQzLTIuMjgsMS41OS0yLjM2LDMuMTItLjA3LDEuMzItLjA1LDIuNjUsMCwzLjk3LjAxLjMxLS4wNC40My0uNC40My0xLjAxLS4wMy0yLjAxLS4wMi0zLjAyLDAtLjM0LDAtLjQ1LS4wNy0uNDQtLjQzLjAyLTEuNzksMC0zLjU3LDAtNS4zNloiLz4KPC9zdmc+';
const imgMenu = 'data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iTGF5ZXJfMSIgZGF0YS1uYW1lPSJMYXllciAxIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDgwIDgwIj4KICA8ZGVmcz4KICAgIDxzdHlsZT4KICAgICAgLmNscy0xIHsKICAgICAgICBmaWxsOiAjZTEyNDJiOwogICAgICAgIHN0cm9rZTogI2ZmZjsKICAgICAgICBzdHJva2UtbGluZWNhcDogcm91bmQ7CiAgICAgICAgc3Ryb2tlLWxpbmVqb2luOiByb3VuZDsKICAgICAgICBzdHJva2Utd2lkdGg6IDNweDsKICAgICAgfQoKICAgICAgLmNscy0yIHsKICAgICAgICBmaWxsOiAjZmZmOwogICAgICAgIHN0cm9rZS13aWR0aDogMHB4OwogICAgICB9CiAgICA8L3N0eWxlPgogIDwvZGVmcz4KICA8cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik0zOS40LDYxLjJjLTEuMi0xLjgtMi41LTMuNy0zLjctNS41LTQuNC02LjYtOC44LTEzLjItMTMuMi0xOS43LS4zLS41LS41LTEuMy0xLjQtMS4xLTEsLjItLjcsMS4xLS43LDEuOCwwLDEyLjEsMCwyNC4xLDAsMzYuMiwwLDEuNS0uNCwxLjktMS45LDEuOS00LjYsMC05LjIsMC0xMy44LDAtMS4yLDAtMS41LS4zLTEuNS0xLjUsMC0yMi4xLDAtNDQuMiwwLTY2LjMsMC0xLjMuNC0xLjYsMS42LTEuNiw1LjQsMCwxMC45LDAsMTYuMywwLDEuMSwwLDEuNy40LDIuMywxLjMsNC43LDgsOS40LDE1LjksMTQuMSwyMy45LDEuNSwyLjYsMi4zLDIuNiwzLjksMCw1LTcuOSwxMC0xNS43LDE0LjktMjMuNi43LTEuMiwxLjUtMS42LDIuOS0xLjYsNS4yLDAsMTAuNCwwLDE1LjUsMCwxLjIsMCwxLjguMiwxLjgsMS42LDAsMjIsMCw0NC4xLDAsNjYuMSwwLDEuNC0uNSwxLjctMS44LDEuNi00LjksMC05LjcsMC0xNC42LDAtMS41LDAtMS44LS40LTEuOC0xLjgsMC0xMi4xLDAtMjQuMywwLTM2LjRzMC0uOSwwLTEuM2MwLS42LjItMS40LS41LTEuNy0uOS0uMy0xLjEuNi0xLjUsMS4yLTUuMyw4LjMtMTAuNiwxNi43LTE1LjksMjUtLjMuNS0uNSwxLjItMS4zLDEuNWgwWiIvPgogIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTM5LjQsNjEuMmMtMS4yLTEuOC0yLjUtMy43LTMuNy01LjUtNC40LTYuNi04LjgtMTMuMi0xMy4yLTE5LjctLjMtLjUtLjUtMS4zLTEuNC0xLjEtMSwuMi0uNywxLjEtLjcsMS44LDAsMTIuMSwwLDI0LjEsMCwzNi4yLDAsMS41LS40LDEuOS0xLjksMS45LTQuNiwwLTkuMiwwLTEzLjgsMC0xLjIsMC0xLjUtLjMtMS41LTEuNSwwLTIyLjEsMC00NC4yLDAtNjYuMywwLTEuMy40LTEuNiwxLjYtMS42LDUuNCwwLDEwLjksMCwxNi4zLDAsMS4xLDAsMS43LjQsMi4zLDEuMyw0LjcsOCw5LjQsMTUuOSwxNC4xLDIzLjksMS41LDIuNiwyLjMsMi42LDMuOSwwLDUtNy45LDEwLTE1LjcsMTQuOS0yMy42LjctMS4yLDEuNS0xLjYsMi45LTEuNiw1LjIsMCwxMC40LDAsMTUuNSwwLDEuMiwwLDEuOC4yLDEuOCwxLjYsMCwyMiwwLDQ0LjEsMCw2Ni4xLDAsMS40LS41LDEuNy0xLjgsMS42LTQuOSwwLTkuNywwLTE0LjYsMC0xLjUsMC0xLjgtLjQtMS44LTEuOCwwLTEyLjEsMC0yNC4zLDAtMzYuNHMwLS45LDAtMS4zYzAtLjYuMi0xLjQtLjUtMS43LS45LS4zLTEuMS42LTEuNSwxLjItNS4zLDguMy0xNi40LDI2LjItMTcuMSwyNi42aDBaIi8+Cjwvc3ZnPg==';

class MicrobitTensorFlow {
    getInfo () {
        return {
            id: 'microbitTensorFlow',
            name: formatMessage({
                id: 'microbitTensorFlow.name',
                default: 'Micro:bit TensorFlow'
            }),
            menuIconURI: imgMenu,
            blockIconURI: imgBLOC,
            blocks: [
                {
                    opcode: 'connect',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'microbitTensorFlow.connect',
                        default: 'Connect to micro:bit'
                    })
                },
                {
                    opcode: 'disconnect',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'microbitTensorFlow.disconnect',
                        default: 'Disconnect from micro:bit'
                    })
                },
                {
                    opcode: 'sendData',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'microbitTensorFlow.sendData',
                        default: 'Send [DATA] to micro:bit'
                    }),
                    arguments: {
                        DATA: {
                            type: ArgumentType.STRING,
                            defaultValue: formatMessage({
                                id: 'microbitTensorFlow.defaultData',
                                default: 'Hello micro:bit'
                            })
                        }
                    }
                },
                {
                    opcode: 'receiveData',
                    blockType: BlockType.REPORTER,
                    text: formatMessage({
                        id: 'microbitTensorFlow.receiveData',
                        default: 'Receive data from micro:bit'
                    })
                },
                {
                    opcode: 'setClasses',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'microbitTensorFlow.setClasses',
                        default: 'Set classes [CLASSES]'
                    }),
                    arguments: {
                        CLASSES: {
                            type: ArgumentType.STRING,
                            defaultValue: 'Class1,Class2'
                        }
                    }
                },
                {
                    opcode: 'collectTrainingData',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'microbitTensorFlow.collectTrainingData',
                        default: 'Collect training data for class [CLASS] for [SECONDS] seconds'
                    }),
                    arguments: {
                        CLASS: {
                            type: ArgumentType.STRING,
                            defaultValue: 'Class1'
                        },
                        SECONDS: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 1
                        }
                    }
                },
                {
                    opcode: 'trainModel',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'microbitTensorFlow.trainModel',
                        default: 'Train model'
                    })
                },
                {
                    opcode: 'predict',
                    blockType: BlockType.REPORTER,
                    text: formatMessage({
                        id: 'microbitTensorFlow.predict',
                        default: 'Predict class for input [INPUT]'
                    }),
                    arguments: {
                        INPUT: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 0
                        }
                    }
                },
                {
                    opcode: 'checkPrediction',
                    blockType: BlockType.BOOLEAN,
                    text: formatMessage({
                        id: 'microbitTensorFlow.checkPrediction',
                        default: 'Is prediction [PREDICTION] equal to class [CLASS]'
                    }),
                    arguments: {
                        PREDICTION: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 0
                        },
                        CLASS: {
                            type: ArgumentType.STRING,
                            defaultValue: 'Class1'
                        }
                    }
                }
            ],
            menus: {},
            translation_map: {
                es: {
                    'microbitTensorFlow.name': 'Micro:bit TensorFlow',
                    'microbitTensorFlow.connect': 'Conectar a la micro:bit',
                    'microbitTensorFlow.disconnect': 'Desconectar de la micro:bit',
                    'microbitTensorFlow.sendData': 'Enviar [DATA] a la micro:bit',
                    'microbitTensorFlow.receiveData': 'Recibir datos de la micro:bit',
                    'microbitTensorFlow.defaultData': 'Hola micro:bit',
                    'microbitTensorFlow.setClasses': 'Establecer clases [CLASSES]',
                    'microbitTensorFlow.collectTrainingData': 'Recoger datos de entrenamiento para la clase [CLASS] por [SECONDS] segundos',
                    'microbitTensorFlow.trainModel': 'Entrenar modelo',
                    'microbitTensorFlow.predict': 'Predecir clase para entrada [INPUT]',
                    'microbitTensorFlow.checkPrediction': 'La predicción [PREDICTION] es igual a la clase [CLASS]'
                },
                en: {
                    'microbitTensorFlow.name': 'Micro:bit TensorFlow',
                    'microbitTensorFlow.connect': 'Connect to micro:bit',
                    'microbitTensorFlow.disconnect': 'Disconnect from micro:bit',
                    'microbitTensorFlow.sendData': 'Send [DATA] to micro:bit',
                    'microbitTensorFlow.receiveData': 'Receive data from micro:bit',
                    'microbitTensorFlow.defaultData': 'Hello micro:bit',
                    'microbitTensorFlow.setClasses': 'Set classes [CLASSES]',
                    'microbitTensorFlow.collectTrainingData': 'Collect training data for class [CLASS] for [SECONDS] seconds',
                    'microbitTensorFlow.trainModel': 'Train model',
                    'microbitTensorFlow.predict': 'Predict class for input [INPUT]',
                    'microbitTensorFlow.checkPrediction': 'Is prediction [PREDICTION] equal to class [CLASS]'
                }
            }
        };
    }

    connect () {
        if (!isConnected) {
            connect();
        }
    }

    disconnect () {
        disconnect();
    }

    sendData (args) {
        sendData(args.DATA);
    }

    async receiveData () {
        const data = await receiveData();
        return data ? data.toString() : '';
    }

    setClasses (args) {
        classNames = args.CLASSES.split(',').map(name => name.trim());
        numClasses = classNames.length;
    }

    async collectTrainingData (args) {
        const classIndex = classNames.indexOf(args.CLASS);
        if (classIndex !== -1) {
            await collectTrainingData(classIndex, args.SECONDS);
        }
    }

    async trainModel () {
        if (numClasses > 0) {
            model = createModel(numClasses);
            await trainModel();
        }
    }

    async predict (args) {
        const result = await predict(args.INPUT);
        return result.toString();
    }

    async checkPrediction(args) {
        const input = parseFloat(args.INPUT);
        const prediction = await predict(input);
        const classIndex = classNames.indexOf(args.CLASS);
        return prediction === classIndex;
    }
}

module.exports = MicrobitTensorFlow;
