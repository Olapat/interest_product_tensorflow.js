let data = DATA; //รับ Data จาก from form-interest-product-export.js
let dataInteger = [];
let productInteger = [];
let chart;

let keys = Object.keys(data);
let labalProduct = ['อุปกรณ์อิเล็กทรอนิกส์', 'สุขภาพและความงาม', 'แฟชั่น', 'ของใช้ในบ้านและเครื่องมือต่างๆ', 'เกมและซอฟแวร์'];
//แทนข้อมูลให้เป็นตัวเลข
for (let key of keys) {                         //วนรูปตามจำนวน keys ของ data
    let val = data[key];                        //เก็บ ค่า ของ data ในแต่ละ key
    let ageInt = parseInt(val.age);             //แปลง string เป็น integer
    let ageFloat = ageInt / 100;                //หารด้วย 100 เพื่อแปลงให้เป็น ทศนิยม ระหว่าง 0 - 1
    //ตรวจสอบ เพศว่าเท่ากับ ชาย หรือไม่ ถ้าใช้ ให้แทนเป็นตัวเลข 0 ถ้าไม่ แทนเป็น 1
    let keySex = val.sex === 'ชาย' ? 0 : 1;
    //ตรวจสอบ สถานะว่าเท่ากับ โสด หรือไม่ ถ้าใช้ ให้แทนเป็นตัวเลข 0 ถ้าไม่ แทนเป็น 1
    let keyStatus = val.status === 'โสด' ? 0 : 1;
    /*ตรวจสอบ สินค้าว่าเท่ากับ อุปกรณ์อิเล็กทรอนิกส์ หรือไม่ ถ้าใช้ ให้แทนเป็นตัวเลข 0 ถ้าไม่ ตรวจสอบอีกว่าเท่ากับ สุขภาพและความงาม 
    หรือไม่ ถ้าใช้ ให้แทนเป็น 1 ตรวจสอบและแทนค่าไปจนกว่าจะครบ */
    let keyProduct = val.product === 'อุปกรณ์อิเล็กทรอนิกส์' ? 0 : val.product === 'สุขภาพและความงาม' ? 1 : 
        val.product === 'แฟชั่น' ? 2 : val.product === 'ของใช้ในบ้านและเครื่องมือต่างๆ' ? 3 : val.product === 'เกมและซอฟแวร์' ? 4 : 
        5 // 5 = ไมสน
    let d = [ageFloat, keySex, keyStatus]       //เก็บข้อมูลที่ แทนด้วยตัวเลข ไว้ใน array d
    dataInteger.push(d)                         //เก็บข้อมูล array d ไว้ใน array dataInteger
    productInteger.push(keyProduct);            //เก็บข้อมูลสิค้าที่แทนด้วยตัวเลข ไว้ไน array productInteger
}


//สร้าง รูปแบบ ข้อมูลเข้า (xs) และ ข้อมูลออก (ys)

//ข้อมูลนำเข้า (input)
let xs = tf.tensor2d(dataInteger);
xs.print();                 //แสดงผล

//ข้อมูลคำตอบ รูปแบบ array1d
let interest1D = tf.tensor1d(productInteger, 'int32');
// interest1D.print();

//ข้อมูลที่ต้องการ (output)
let ys = tf.oneHot(interest1D, 5).cast('float32');
ys.print(); 

interest1D.dispose();       //คืนค่า (ไม่จำ)


//สร้าง model NN
let model = tf.sequential();

//สร้าง layers hidden
const hidden = tf.layers.dense({
    units: 25,                  //จำนวนโหนด
    inputShape: [3],            //จำนวนค่าที่รับเข้ามา (input)
    activation: 'relu',         //สูตรสมการที่ใช้
    useBias: true               //ใช้ค่าเบี่ยงเบน เพื่อช่วยในการคำนวณ
});

//สร้าง layers output
const output = tf.layers.dense({
    units: 5,                   //จำนวนโหนด
    activation: 'softmax'       //สูตรสมการที่ใช้
});

//เพิ่ม layers ที่สร้าง ลงใน model
model.add(hidden);
model.add(output);



let optimizer = tf.train.sgd(0.06);      //กำหนดค่า sgb (ค่าละเอียดในการเรียนรู้) ในตัวแปล optimizer
model.compile({
    optimizer: optimizer,               //นำตัวแปล optimizer มาใช้
    loss: 'categoricalCrossentropy',    //กำหนด ค่า loss (ค่าสูญเสีย)
});

// function การเรียนรู้ หรือ ทดสอบ
async function train() { 
    await model.fit(xs, ys, {                       //ใส่ค่า input, output
        epochs: 250,                                //จำนวนรอบในการเรียนรู้
        callbacks: {                                //กำหนดการทำงานในระหว่างการเรียนรู้
            onEpochEnd: (e, l) => {                 //เมื่อเรียนรู้เสร็จในแต่ระรอบ
                console.log(e);                     //แสดงจำนวนรอบผ่าน console
                console.log(l.loss.toFixed(5));     //แสดงค่าความสูญเสิยผ่าน console ในรูปแบบทศนิยม 5 หลัก
            },
            onTrainEnd: () => {                     //เมื่อเรียนรู้สำเร็จ
                console.log("เทรดเสร็จแล้ว");          //แสดงข้อความผ่าน console
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
