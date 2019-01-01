let data = DATA; //รับ Data จาก from form-interest-product-export.js
let dataInteger = [];
let interest = [];
let chart;

let keys = Object.keys(data);
let labalProduct = ['อุปกรณ์อิเล็กทรอนิกส์', 'สุขภาพและความงาม', 'แฟชั่น', 'ของใช้ในบ้านและอุปกรณ์', 'เกมและซอฟแวร์'];

//แทนข้อมูลให้เป็นตัวเลข
for (let key of keys) {
    let val = data[key];
    let ageInt = parseInt(val.age)
    let ageFloat = ageInt / 100
    let keySex = val.sex === 'ชาย' ? 0 : 1 
    let keyStatus = val.status === 'โสด' ? 0 : 1
    let keyProduct = val.product === 'อุปกรณ์อิเล็กทรอนิกส์' ? 0 : val.product === 'สุขภาพและความงาม' ? 1 : 
        val.product === 'แฟชั่น' ? 2 : val.product === 'ของใช้ในบ้านและอุปกรณ์' ? 3 : val.product === 'เกมและซอฟแวร์' ? 4 : 
        5 // 5 = ไมสน
    let d = [ageFloat, keySex, keyStatus]
    dataInteger.push(d)
    interest.push(keyProduct);
}
console.log(keys.length)
// console.log(dataInteger);
// console.log(interest);

//สร้าง รูปแบบ ข้อมูลเข้า (xs) และ ข้อมูลออก (ys)

//ข้อมูลนำเข้า (input)
let xs = tf.tensor2d(dataInteger);
xs.print();                 //แสดงผล

//ข้อมูลคำตอบ รูปแบบ array1d
let interest1D = tf.tensor1d(interest, 'int32');
// interest1D.print();

//ข้อมูลที่ต้องการ (output)
let ys = tf.oneHot(interest1D, 5).cast('float32');
ys.print(); 

interest1D.dispose();       //คืนค่า (ไม่จำ)


//สร้าง model NN
let model = tf.sequential();

//สร้าง layers hidden
const hidden = tf.layers.dense({
    units: 8,                  //จำนวนโหนด
    inputShape: [3],            //จำนวนค่าที่รับเข้ามา (input)
    activation: 'sigmoid'       //สูตรสมการที่ใช้
});

//สร้าง layers output
const output = tf.layers.dense({
    units: 5,               //จำนวนโหนด
    activation: 'softmax'   //สูตรสมการที่ใช้
});

//เพิ่ม layers ที่สร้าง ลงใน model
model.add(hidden);
model.add(output);



let optimizer = tf.train.sgd(0.15);      //กำหนดค่า sgb (ค่าละเอียดในการเรียนรู้) ในตัวแปล optimizer
model.compile({
    optimizer: optimizer,               //นำตัวแปล optimizer มาใช้
    loss: 'categoricalCrossentropy',    //กำหนด ค่า loss (ค่าสูญเสีย)
});

// function การเรียนรู้ หรือ ทดสอบ
async function train() { 
    await model.fit(xs, ys, {                   //ใส่ค่า input, output
        // shuffle: true,                              //สลับข้อมูล 
        // validationSplit: 0.01,                  //กำหนดค่าความสูญเสียของข้อมูล
        batchSize: 70,
        epochs: 800,                           //จำนวนรอบในการเรียนรู้
        callbacks: {                            //กำหนดการทำงานในระหว่างการเรียนรู้
            onEpochEnd: (e, l) => {             //เมื่อเรียนรู้เสร็จในแต่ระรอบ
                console.log(e, '|', l.loss);
            },
            onTrainEnd: () => {                 //เมื่อเรียนรู้สำเร็จ
                console.log("เทรดเสร็จแล้ว");
            }
        }
    });
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
    tf.tidy(() => {
        let results = model.predict(dataTest);      //ส่งข้อมูลให้ AI วิเคราะห์
        results.print();                            //ผลลัพธ์ รูปแบบ arrayTensor
        let resultsDataSync = results.dataSync();   //ผลลัพธ์ รูปแบบ float 32
        chart = Array.from(resultsDataSync);        //ผลลัพธ์ รูปแบบ array ปกติ
        chartMaxToMin = chart.slice();              //clone เพื่อนำไปเรียงลำดับ
        chartMaxToMin.sort(function(a, b){          //เรียงจาก มาก - น้อย
            return b-a
        });
    });
    
    //ส่งผลลัพธ์ chart ไปให้ function disPlayResults เพื่อแสดงออกหน้าเว็บ
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
