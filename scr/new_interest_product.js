let data = DATA; //from form-interest-product-export.js
let dataInput = [];
let dataOutput = [];  

//เตรียมข้อมูล แปลงเป็นตัวเลข / Input Output 
const keys = Object.keys(data);     //ดึงเอา key จากข้อมูล
const numExamples = keys.length;    //จำนวนข้อมูลทั้งหมด
tf.util.shuffle(keys);              //สลับข้อมูล
const labalProduct = ['อุปกรณ์อิเล็กทรอนิกส์', 'สุขภาพและความงาม',  //หมวดหมู่สินค้า
     'แฟชั่น', 'ของใช้ในบ้านและเครื่องมือต่างๆ', 'เกมและซอฟแวร์'];
const numClass = labalProduct.length;       //จำนวนหมวดหมู่สินค้า
//วนรูปเพื่อนำข้อมูลแต่ละตัวมาแปลงเป็นตัวเลข และแยกออกเป็น Input, Output
for (const key of keys) {               //วนรูปตามจำนวนข้อมูล
    let val = data[key];                //เก็บข้อมูลของ key นั้นๆ ไว้ใน val
    let ageInt = parseInt(val.age);      //
    let ageFlost = ageInt / 10;             
    // let agetoFixed = ageFlost.toFixed(1);
    // let age = parseFloat(agetoFixed);
    let keySex = val.sex === 'ชาย' ? 0 : 1;
    let keyStatus = val.status === 'โสด' ? 0 : 1;
    //ตรวจสอบ / แทนค่า และ เก็บไว้ใน keyProduct
    let keyProduct = val.product === 'อุปกรณ์อิเล็กทรอนิกส์' ? 0    //เป็น"อุปกรณ์อิเล็กทรอนิกส์"ให้แทนเป็น 0
        : val.product === 'สุขภาพและความงาม' ? 1                //เป็น"สุขภาพและความงาม"ให้แทนเป็น 1
        : val.product === 'แฟชั่น' ? 2                           //เป็น"สุขภาพและความงาม"ให้แทนเป็น 2
        : val.product === 'ของใช้ในบ้านและเครื่องมือต่างๆ' ? 3        //เป็น"ของใช้ในบ้านและเครื่องมือต่างๆ"ให้แทนเป็น 3
        : val.product === 'เกมและซอฟแวร์' ? 4                   //เป็น"เกมและซอฟแวร์"ให้แทนเป็น 4
        : 5 // 5 = ไม่ทราบข้อมูล
    let d = [ageFlost, keySex, keyStatus]
    dataInput.push(d)
    dataOutput.push(keyProduct);
}

const numTestExamples = Math.round(numExamples * 0.8);      //กำหนดขนาดของข้อมูลไว้สำหรับ test 80%
const numTrainExamples = numExamples - numTestExamples;     //กำหนดขนาดของข้อมูลที่เหลือไว้สำหรับ train
const xDims = dataInput[0].length;                          //กำหนดขนาด Input (3)
// console.log(interest);

//ออกแบบ tensor xs, ys
let xs = tf.tensor2d(dataInput);                            //สร้าง Input เป็น tensor2d
// xs.print();

let interest1D = tf.tensor1d(dataOutput, 'int32');          //สร้าง Output เป็น tensor1d
// interest1D.print();

let ys = tf.oneHot(interest1D, numClass).cast('float32');   //สร้าง Output เป็นแบบ tensorOneHot
// ys.print();
interest1D.dispose();

//แบ่งข้อมูลไว้สำหรับ Train และ Test
const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);                 //แบ่งข้อมูล Input สำหรับ Train
const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);    //แบ่งข้อมูล Input สำหรับ Test
const yTrain = ys.slice([0, 0], [numTrainExamples, numClass]);              //แบ่งข้อมูล Output สำหรับ Train
const yTest = ys.slice([0, 0], [numTestExamples, numClass]);                //แบ่งข้อมูล Output สำหรับ Test

let model = tf.sequential();        //สร้าง model
//สร้าง hidden layer
const hidden = tf.layers.dense({
    units: 50,                      //จำนวนโหนด
    inputShape: [xDims],            //จำนวนค่าที่รับเข้ามา (input) 
    activation: 'sigmoid'           //function ที่ใช้ในการคำนวณ
});

//สร้าง output layer
const output = tf.layers.dense({
    units: numClass,                //จำนวนโหนด
    activation: 'softmax'           //function ที่ใช้ในการคำนวณ
});

//เพิ่ม layers ที่สร้างลงไปใน model
model.add(hidden);          
model.add(output);
// model.summary();

const optimizer = tf.train.adam(0.01);      //กำหนดความละเอียดในการเรียนรู้ Train
model.compile({
    optimizer: optimizer,                   //ใส่ค่าความละเอียดในการเรียนรู้ที่กำหนดไว้
    loss: 'categoricalCrossentropy',        //กำหนด loss function ที่ใช้ในการเปรียบเทียบผลลัพธ์
    metrics: ['accuracy']                   //เพิ่มค่าความแม่นยำไว้ใน metrics 
});

async function train() {
    //เริ่ม Train โดยระบุรายละเอียด ดังนี้
    await model.fit(xTrain, yTrain, {               //ระบุข้อมูลสำหรับ Train
        epochs: 500,                                //ระบุจำนวนรอบ
        validationData: [xTest, yTest],             //ระบุข้อมูลสำหรับ Test
        callbacks: {                                //เพิ่มการทำงานในระหว่าง Train
            //ทำงานเมื่อ Train จบในแต่ละรอบ
            onEpochEnd: (e, l) => {                 
                console.log(e);                     //แสดงจำนวนรอบผ่าน console
                console.log(l.loss, '|', l.acc);    //แสดงค่า loss และ acc ผ่าน console
            },
            //ทำงานเมื่อ Train ครบแล้ว
            onTrainEnd: () => {
                console.log("เทรดเสร็จแล้ว");          //แสดงข้อความเตือนผ่าน console
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
    let getKeySex = getSex === 'ชาย' ? 0 : 1 ;
    let getKeyStatus = getStatus === 'โสด' ? 0 : 1;
    let getAgeFloat = getAgeInt / 10;
    let agetoFixed = getAgeFloat.toFixed(1);
    let getage = parseFloat(agetoFixed);
    let test2D = [
        [getage, getKeySex, getKeyStatus]
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
