let data = DATA;
let dataInteger = [];
let rateProduct = [];
let labalProduct;
let keys = Object.keys(data);
tf.util.shuffle(keys);

const numData = keys.length;
console.log(numData);

tf.tidy(()=> {
    for (const key of keys) {
        let value = data[key];
        let ageInt = parseInt(value.age);
        let ageFloat = ageInt / 100;
        let keySex = value.sex === 'ชาย' ? 0 : 1;
        let keyStatus = value.status === 'โสด' ? 0 : 1;
        dataInteger.push([ageFloat, keySex, keyStatus]);
        let valueP = Object.values(data[key].productRate);
        let newVaP = [];
        valueP.forEach((va) => {
            let h = parseInt(va) / 5;
            let hh = h.toFixed(1);
            let hhh = parseFloat(hh);
            newVaP.push(hhh);
        });
        rateProduct.push(newVaP);
        labalProduct = Object.keys(data[key].productRate);
    }
}); 


// console.log(dataInteger);
// console.log(rateProduct);
// console.log(labalProduct);

const numTestData = Math.round(numData * 0.8);
const numTrainData = numData - numTestData;
const xDim = dataInteger[0].length;
const numClass = labalProduct.length;
console.log(numTestData);

//สร้าง รูปแบบ ข้อมูลเข้า (xs) และ ข้อมูลออก (ys)

//ข้อมูลนำเข้า (input)

let xs = tf.tensor2d(dataInteger);
xs.print();                 //แสดงผล

let ys = tf.tensor2d(rateProduct);
ys.print();

const xTrain = xs.slice([0, 0], [numTrainData, xDim]);
const xTest = xs.slice([numTrainData, 0], [numTestData, xDim]);
const yTrain = ys.slice([0, 0], [numTrainData, numClass]);
const yTest = ys.slice([0, 0], [numTestData, numClass]);

//สร้าง model NN
let model = tf.sequential();

//สร้าง layers hidden
const hidden = tf.layers.dense({
    units: 50,                  //จำนวนโหนด
    inputShape: [xDim],            //จำนวนค่าที่รับเข้ามา (input)
    activation: 'sigmoid',         //สูตรสมการที่ใช้
    // useBias: true               //ใช้ค่าเบี่ยงเบน เพื่อช่วยในการคำนวณ
});

//สร้าง layers output
const output = tf.layers.dense({
    units: numClass,     //จำนวนโหนด
    activation: 'softmax',          //สูตรสมการที่ใช้
});

//เพิ่ม layers ที่สร้าง ลงใน model
model.add(hidden);
model.add(output);

let optimizer = tf.train.adam(0.01);      //กำหนดค่า sgd (ค่าละเอียดในการเรียนรู้) ในตัวแปล optimizer
model.compile({
    optimizer: optimizer,                   //นำตัวแปล optimizer มาใช้
    loss: 'categoricalCrossentropy',         //กำหนด ค่า loss (ค่าสูญเสีย) กำหนดตัวชี้วัดความแตกต่างกันระหว่างเอาท์พุทที่ได้ระหว่างการเรียนรู้ กับ ผลลัพธ์(ที่เรากำหนดให้ ใน ys)
    metrics: ['accuracy']                  //กำหนด ค่า accuracy (ค่าความแม่นยำ)
});

// function การเรียนรู้ หรือ ทดสอบ
async function train() {
    await model.fit(xTrain, yTrain, {                       //ใส่ค่า input, output
        epochs: 500,                                //จำนวนรอบในการเรียนรู้
        validationData: [xTest, yTest],
        callbacks: {                                //กำหนดการทำงานในระหว่างการเรียนรู้
            onEpochEnd: (e, l) => {                 //เมื่อเรียนรู้เสร็จในแต่ระรอบ
                console.log(e);                     //แสดงจำนวนรอบผ่าน console
                console.log(l.loss.toFixed(5), ':', l.acc);     //แสดงค่าความสูญเสิยผ่าน console ในรูปแบบทศนิยม 5 ตำแหน่ง
            },
            onTrainEnd: () => {                     //เมื่อเรียนรู้สำเร็จ
                console.log("เทรดเสร็จแล้ว");          //แสดงข้อความผ่าน console
            }
        }
    });
    // console.log(loss);
}

// train();

//function รับข้อมูลจาก index.html
function getData() {
    let getAge = document.getElementById('age').value; 
    let getSex = document.getElementById('sex').value;
    let getStatus = document.getElementById('status').value;

    // แทนข้อมูลให้เป็นตัวเลข เช่นเดียวกับการสร้างข้อมูลให้ AI เรียนรู้
    let getAgeInt = parseInt(getAge);               //แปลง string เป็น int 
    let getKeySex = getSex === 'ชาย' ? 0 : 1 ; 
    let getKeyStatus = getStatus === 'โสด' ? 0 : 1;

    let arrayData = [                               //ยัดข้อมูลที่แปลง ลง array 
        [getAgeInt / 100, getKeySex, getKeyStatus]
    ]

    let dataTensor2D = tf.tensor2d(arrayData);      //แปลงข้อมูลที่รับมาในเป็นรูบแปป tensor2d
    toAITest(dataTensor2D);                         //ส่งค่าไปที่ function toAITest เพื่อให้ AI วิเคราะห์
}


//function ในการ ให้ AI วิเคราะห์หาผลลัพธ์ จากข้อมูลที่เราให้ AI เรียนรู้
function toAITest(dataTest) {
    let chartMaxToMin = [];
    tf.tidy(() => {                                 //tidy เป็น fn ที่จะคืนค่า memory หลังจากทำงานเสร็จ
        let results = model.predict(dataTest);      //ส่งข้อมูลให้ AI วิเคราะห์
        results.print();                            //แสดงผลลัพธ์ รูปแบบ arrayTensor
        let resultsDataSync = results.dataSync();   //เก็บค่าผลลัพธ์ รูปแบบ float 32
        chart = Array.from(resultsDataSync);        //แปลงผลลัพธ์ในเป็นรูปแบบ array ปกติ
        chartMaxToMin = chart.slice();              //clone array เพื่อนำไปเรียงลำดับ
        chartMaxToMin.sort(function(a, b){          //เรียงจาก มาก - น้อย
            return b-a
        });
    });
    
    //ส่งผลลัพธ์ ที่ผ่านการเรียงลำดับแล้ว ไปให้ function disPlayResults เพื่อแสดงออกหน้าเว็บ
    disPlayResults(chartMaxToMin);      
}

// function แสดงผลลัพธ์ บน html ตามลำดับ
function disPlayResults(resultsChart) {
    let arrDisplay = [];
    resultsChart.forEach((va) => {
        //หาค่า index เพื่อแสดงข้อมูล ในรูปแบบ string
        let indexLabalProduct = chart.indexOf(va);
        //นำ index ที่ได้ มาดึงข้อมูลสินค้า
        let displaylabalProduct = labalProduct[indexLabalProduct];  

        let persen = va * 100 // แปลงจาก ทศนิยม เป็น จำนวนเต็ม
        let displayChartPersen = persen.toFixed(2);     //ปรับเป็นทศนิยม 2 ตำแหน่ง   
        
        arrDisplay.push(                                //เพิ่ม element (view) เข้าไป array arrDisplay
            `<tr>
                <td>${displaylabalProduct}</td>
                <td>${displayChartPersen}</td>
            </tr>`
        );
        //นำ arrDisplay มาแสดงใน HTML
        document.getElementById('tex').innerHTML = arrDisplay.join('');
    });
}
