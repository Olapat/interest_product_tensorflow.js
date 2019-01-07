let dataJ = DATA; //from form-interest-product-export.js
let dataAll = [];
let interest = [];  
let keys = Object.keys(dataJ);
const numExamples = keys.length;
tf.util.shuffle(keys);
let labalProduct = ['อุปกรณ์อิเล็กทรอนิกส์', 'สุขภาพและความงาม', 'แฟชั่น', 'ของใช้ในบ้านและเครื่องมือต่างๆ', 'เกมและซอฟแวร์'];
let numClass = labalProduct.length;
for (let key of keys) {
    let val = dataJ[key];
    let ageInt = parseInt(val.age)
    let ageFlost = ageInt / 10;
    let agetoFixed = ageFlost.toFixed(1);
    let age = parseFloat(agetoFixed);
    let keySex = val.sex === 'ชาย' ? 0 : 0.1 
    let keyStatus = val.status === 'โสด' ? 0 : 0.2
    let keyProduct = val.product === 'อุปกรณ์อิเล็กทรอนิกส์' ? 0 
        : val.product === 'สุขภาพและความงาม' ? 1 
        : val.product === 'แฟชั่น' ? 2 
        : val.product === 'ของใช้ในบ้านและเครื่องมือต่างๆ' ? 3 
        : 4 // 4 = เกมและซอฟแวร์
    let sex = age * keySex;
    let status = age * keyStatus;
    let d = [age, sex, status]
    dataAll.push(d)
    interest.push(keyProduct);
}

const numTestExamples = Math.round(numExamples * 0.8);
const numTrainExamples = numExamples - numTestExamples;
const xDims = dataAll[0].length;
// console.log(interest);

//ออกแบบ tensor xs, ys
let xs = tf.tensor2d(dataAll);
// xs.print();

let interest1D = tf.tensor1d(interest, 'int32');
// interest1D.print();

let ys = tf.oneHot(interest1D, numClass).cast('float32');
// ys.print();
interest1D.dispose();

const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
const yTrain = ys.slice([0, 0], [numTrainExamples, numClass]);
const yTest = ys.slice([0, 0], [numTestExamples, numClass]);

const xTrains = [];
const yTrains = [];
const xTests = [];
const yTests = [];
for (let i = 0; i < numClass.length; ++i) {
    xTrains.push(xTrain);
    yTrains.push(yTrain);
    xTests.push(xTest);
    yTests.push(yTest);
}

let model = tf.sequential();

const hidden = tf.layers.dense({
    units: 50,
    inputShape: [xDims],
    activation: 'sigmoid'
});

const output = tf.layers.dense({
    units: numClass,
    activation: 'softmax'
});

model.add(hidden);
model.add(output);
model.summary();

let optimizer = tf.train.adam(0.01);
model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});

async function train() {
    let acc;
    await model.fit(xTrain, yTrain, {
        epochs: 1000,
        validationData: [xTest, yTest], 
        callbacks: {
            onEpochEnd: (e, l) => {
                console.log(e, '|', l.acc);
                // acc = l.acc;
            },
            onTrainEnd: () => {
                console.log("เทรดเสร็จแล้ว");
                // console.log(acc);
            }
        }
    });
}

// train();

function getQQ() {
    let getAge = document.getElementById('age').value;
    let getSex = document.getElementById('sex').value;
    let getStatus = document.getElementById('status').value;

    let getAgeInt = parseInt(getAge);
    let getKeySex = getSex === 'ชาย' ? 0 : 0.1 ;
    let getKeyStatus = getStatus === 'โสด' ? 0 : 0.2;
    let getAgeFloat = getAgeInt / 10;
    let agetoFixed = getAgeFloat.toFixed(1);
    let getage = parseFloat(agetoFixed);
    let sex = getage * getKeySex;
    let status = getage * getKeyStatus;
    let test2D = [
        [getage, sex, status]
    ]

    let totensor2D = tf.tensor2d(test2D);
    // console.log(totensor2D);
    toAi(totensor2D);
}

function toAi(dataTest) {
    tf.tidy(() => {
        let results = model.predict(dataTest);
        results.print();
        let resultsDataSync = results.dataSync();
        let chat = Array.from(resultsDataSync);
        // let argMax = results.argMax(1);
        // let indexProductOfTest =  argMax.dataSync();
        // let topProduct = labalProduct[indexProductOfTest];
        // console.log(topProduct);
        let chatMaxToMin = chat.slice(); //clone

        chatMaxToMin.sort(function(a, b){ //เรียง ม - น
            return b-a
        });
        // console.log(chat);
        // console.log(chatMaxToMin);
        let displayChatLabel = [];
        let displayChatPersen = [];
        chatMaxToMin.forEach((va) => {
            let indexLabalProduct = chat.indexOf(va)
            let persen = va * 100
            displayChatLabel.push('<b>'+labalProduct[indexLabalProduct]+'</b> <br/>');
            displayChatPersen.push(persen.toFixed(2)+'<br/>');     
            let displayLabel = document.getElementById('label');
            let displayPersen = document.getElementById('persen');
            displayLabel.innerHTML = displayChatLabel.join('')
            displayPersen.innerHTML = displayChatPersen.join('')
        });
    }) 
}
