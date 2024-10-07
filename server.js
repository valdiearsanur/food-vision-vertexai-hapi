// npm install @hapi/hapi @hapi/inert @hapi/vision @hapi/cookie dotenv tensorflow tfjs-node

'use strict';

const Hapi = require('@hapi/hapi');
const Inert = require('@hapi/inert');
const Vision = require('@hapi/vision');
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const { google } = require('googleapis');
const { ClientOptions } = require('google-api-core');

const PORT = process.env.PORT || 3000;
const PROJECT = "bangkit-academy-437808";
const REGION = "us-central1";
const MODEL_NAME = "food_vision";
const CLASSES = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi'];

const init = async () => {
    const server = Hapi.server({
        port: PORT,
        host: 'localhost'
    });

    await server.register([Inert, Vision]);

    server.route({
        method: 'POST',
        path: '/predict',
        options: {
            payload: {
                output: 'stream',
                parse: true,
                allow: 'multipart/form-data'
            }
        },
        handler: async (request, h) => {
            const { payload } = request;
            const file = payload.file;
            const buffer = await streamToBuffer(file);

            const image = await loadAndPrepImage(buffer);
            const prediction = await makePrediction(image);

            return h.response(prediction).code(200);
        }
    });

    await server.start();
    console.log(`Server running on ${server.info.uri}`);
};

// Utility function to convert stream to buffer
const streamToBuffer = (stream) => {
    return new Promise((resolve, reject) => {
        const chunks = [];
        stream.on('data', chunk => chunks.push(chunk));
        stream.on('end', () => resolve(Buffer.concat(chunks)));
        stream.on('error', reject);
    });
};

// Load and preprocess image
const loadAndPrepImage = async (buffer) => {
    const imageTensor = tf.node.decodeImage(buffer, 3);
    const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);
    const expandedImage = resizedImage.expandDims(0);
    return expandedImage;
};

// Make prediction using Google Cloud ML Engine
const makePrediction = async (image) => {
    const clientOptions = new ClientOptions();
    const modelPath = `projects/${PROJECT}/models/${MODEL_NAME}`;
    const ml = google.ml({
        version: 'v1',
        auth: new google.auth.GoogleAuth({
            keyFile: 'bangkit-academy-437808.json',
            scopes: ['https://www.googleapis.com/auth/cloud-platform']
        }),
        clientOptions
    });

    const instances = image.arraySync();
    const request = ml.projects.predict({
        name: modelPath,
        requestBody: {
            instances: instances
        }
    });

    const response = await request;
    const preds = response.data.predictions;
    const predClass = CLASSES[tf.argMax(preds[0]).dataSync()[0]];
    const predConf = tf.max(preds[0]).dataSync()[0];

    return {
        predClass,
        predConf
    };
};

process.on('unhandledRejection', (err) => {
    console.log(err);
    process.exit(1);
});

init();
