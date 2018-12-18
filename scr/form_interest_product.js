let dataJ = d; //from form-interest-product-export.js
let dataJson;
let dataAll = [];
let interest = [];  
let keys = Object.keys(dataJ);
let labalProduct = ['อุปกรณ์คอม', 'อาหารเสริม', 'แฟชั่น', 'ของใช้ในบ้าน'];
for (let key of keys) {
    let val = dataJ[key];
    let ageInt = parseInt(val.age)
    let ageFlost = ageInt / 60;
    let keySex = val.sex === 'ชาย' ? 0 : 1 
    let keyLW = val.l_or_w === 'เรียน' ? 0 : 1
    let keyStatus = val.status === 'โสด' ? 0 : 1
    let keyProduct = val.product === 'อุปกรณ์คอม' ? 0 : 
    val.product === 'อาหารเสริม' ? 1 : 
    val.product === 'แฟชั่น' ? 2 : 3 // 3 = ของใช้ในบ้าน
    let d = [ageFlost, keySex, keyLW, keyStatus]
    dataAll.push(d)
    interest.push(keyProduct);
}

// console.log(dataAll);
// console.log(interest);

//ออกแบบ tensor xs, ys
let xs = tf.tensor2d(dataAll);
// xs.print();

let interest1D = tf.tensor1d(interest, 'int32');
// interest1D.print();

let ys = tf.oneHot(interest1D, 4).cast('float32');
// ys.print();
interest1D.dispose();

let model = tf.sequential();

const hidden = tf.layers.dense({
    units: 4,
    inputShape: [4],
    activation: 'sigmoid'
});

const output = tf.layers.dense({
    units: 4,
    activation: 'softmax'
});

model.add(hidden);
model.add(output);

let optimizer = tf.train.sgd(0.7);
model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});

async function train() {
    await model.fit(xs, ys, {
        validationSplit: 0.06,
        epochs: 1000, 
        callbacks: {
            onEpochEnd: (e, l) => {
                console.log(e, ' ', l.loss);
            },
            onTrainEnd: () => {
                console.log("เทรดเสร็จแล้ว");
            }
        }
    });
}

// train();

function getQQ() {
    let getAge = document.getElementById('age').value;
    let getSex = document.getElementById('sex').value;
    let getStatus = document.getElementById('status').value;
    let getLorW =  document.getElementById('l_or_w').value;

    let getAgeInt = parseInt(getAge);
    let getKeySex = getSex === 'ชาย' ? 0 : 1 ;
    let getKeyLW = getLorW === 'เรียน' ? 0 : 1;
    let getKeyStatus = getStatus === 'โสด' ? 0 : 1;

    let test2D = [
        [getAgeInt / 60, getKeySex, getKeyLW, getKeyStatus]
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
