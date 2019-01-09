let data = DATA; //จาก data-set/form-interest-product-data-set-export.js
let dataInput = [];
let dataOutput = [];  

//เตรียมข้อมูล แปลงเป็นตัวเลข / Input Output 
const keys = Object.keys(data);     //ดึงเอา key จากข้อมูล
const numData = keys.length;    //จำนวนข้อมูลทั้งหมด
tf.util.shuffle(keys);              //สลับข้อมูล
const labalProduct = ['อุปกรณ์อิเล็กทรอนิกส์', 'สุขภาพและความงาม',  //หมวดหมู่สินค้า
     'แฟชั่น', 'ของใช้ในบ้านและเครื่องมือต่างๆ', 'เกมและซอฟแวร์'];
const numClass = labalProduct.length;       //จำนวนหมวดหมู่สินค้า
//วนรูปเพื่อนำข้อมูลแต่ละตัวมาแปลงเป็นตัวเลข และแยกออกเป็น Input, Output
for (const key of keys) {               //วนรูปตามจำนวนข้อมูล
    let val = data[key];                //เก็บข้อมูลของ key นั้นๆ ไว้ใน val
    let ageInt = parseInt(val.age);      //
    let ageFlost = ageInt / 10;
    let keySex = val.sex === 'ชาย' ? 0 : 1;
    let keyStatus = val.status === 'โสด' ? 0 : 1;
    //ตรวจสอบ / แทนค่า และ เก็บไว้ใน keyProduct
        //"อุปกรณ์อิเล็กทรอนิกส์" แทน 0
    let keyProduct = val.product === 'อุปกรณ์อิเล็กทรอนิกส์' ? 0
        //"สุขภาพและความงาม" แทน 1 
        : val.product === 'สุขภาพและความงาม' ? 1
        //"แฟชั่น" แทน 2              
        : val.product === 'แฟชั่น' ? 2
        //"ของใช้ในบ้านและเครื่องมือต่างๆ" แทน 3
        : val.product === 'ของใช้ในบ้านและเครื่องมือต่างๆ' ? 3
        //"เกมและซอฟแวร์" แทน 4
        : val.product === 'เกมและซอฟแวร์' ? 4
        : 5 // 5 = ไม่ทราบข้อมูล
    let d = [ageFlost, keySex, keyStatus]
    dataInput.push(d)
    dataOutput.push(keyProduct);
}

console.log(numData);

const numTestData = Math.round(numData * 0.8);      //กำหนดขนาดของข้อมูลไว้สำหรับ test 80%
const numTrainData = numData - numTestData;     //กำหนดขนาดของข้อมูลที่เหลือไว้สำหรับ train
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
const xTrain = xs.slice([0, 0], [numTrainData, xDims]);                 //แบ่งข้อมูล Input สำหรับ Train
const xTest = xs.slice([numTrainData, 0], [numTestData, xDims]);    //แบ่งข้อมูล Input สำหรับ Test
const yTrain = ys.slice([0, 0], [numTrainData, numClass]);              //แบ่งข้อมูล Output สำหรับ Train
const yTest = ys.slice([0, 0], [numTestData, numClass]);                //แบ่งข้อมูล Output สำหรับ Test

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
    metrics: ['accuracy']                   //เพิ่มค่าความแม่นยำไว้ใน metrics (acc)
});

async function train() {
    //เริ่ม Train โดยระบุรายละเอียด ดังนี้
    await model.fit(xTrain, yTrain, {       //ระบุข้อมูลสำหรับ Train
        epochs: 1,                          //ระบุจำนวนรอบ
        validationData: [xTest, yTest],     //ระบุข้อมูลสำหรับ Test
        callbacks: {                        //เพิ่มการทำงานในระหว่าง Train
            //ทำงานเมื่อ Train จบในแต่ละรอบ
            onEpochEnd: (epochs, log) => {
                //แสดงจำนวนรอบผ่าน console                 
                console.log("รอบที่: ", epochs);
                //แสดงค่า loss และ acc ผ่าน console
                console.log("loss: ",log.loss, "|","acc: ", log.acc);    
            },
            //ทำงานเมื่อ Train ครบแล้ว
            onTrainEnd: () => {
                //แสดงข้อความเตือนผ่าน console
                console.log("เทรดเสร็จแล้ว");  
            }
        }
    });
}

// train();

function getData() {
    let getAge = document.getElementById('age').value;
    let getSex = document.getElementById('sex').value;
    let getStatus = document.getElementById('status').value;

    let getAgeInt = parseInt(getAge);
    let getKeySex = getSex === 'ชาย' ? 0 : 1 ;
    let getKeyStatus = getStatus === 'โสด' ? 0 : 1;
    let getAgeFloat = getAgeInt / 10;
    let test2D = [
        [getAgeFloat, getKeySex, getKeyStatus]
    ]

    let totensor2D = tf.tensor2d(test2D);
    // console.log(totensor2D);
    toAITest(totensor2D);
}

//function ในการ ให้ AI วิเคราะห์หาผลลัพธ์ จากข้อมูลที่เราให้ AI เรียนรู้
function toAITest(dataTest) {
    let chartMaxToMin = [];
    tf.tidy(() => {  //tidy เป็น fn ที่จะคืนค่า memory หลังจากทำงานเสร็จ
        let results = model.predict(dataTest);      //ส่งข้อมูลให้ AI วิเคราะห์

        //แสดงผลลัพธ์ รูปแบบ arrayTensor ผ่าน console
        results.print();

        //เก็บค่าผลลัพธ์ รูปแบบ float 32
        let resultsDataSync = results.dataSync();

        //แปลงผลลัพธ์ในเป็นรูปแบบ array ปกติ
        chart = Array.from(resultsDataSync);
        
        //clone array เพื่อนำไปเรียงลำดับ
        chartMaxToMin = chart.slice();
        //เรียงจาก มาก - น้อย
        chartMaxToMin.sort(function(a, b){          
            return b-a
        });
    });
    
    /*ส่งผลลัพธ์ ที่ผ่านการเรียงลำดับแล้ว ไปให้ 
    function disPlayResults เพื่อแสดงออกหน้าเว็บ */
    displayResults(chartMaxToMin);      
}

// function แสดงผลลัพธ์ บน html ตามลำดับ
function displayResults(resultsChart) {
    let arrDisplay = [];
    resultsChart.forEach((va) => {
        //หาค่า index เพื่อแสดงข้อมูล ในรูปแบบ string
        let indexLabelProduct = chart.indexOf(va);
        
        //นำ index ที่ได้ มาดึงข้อมูลสินค้า
        let displaylabalProduct = labalProduct[indexLabelProduct];  

        let persen = va * 100 // แปลงจาก ทศนิยม เป็น จำนวนเต็ม
        let displayChartPersen = persen.toFixed(2); //ปรับเป็นทศนิยม 2 ตำแหน่ง   
        
        arrDisplay.push(    //เพิ่ม element (view) เข้าไป array arrDisplay
            `<tr>
                <td>${displaylabalProduct}</td>
                <td>${displayChartPersen}</td>
            </tr>`
        );
        //นำ arrDisplay มาแสดงใน HTML
        document.getElementById('ChartProduct').innerHTML = arrDisplay.join('');
    });
}
