let data = DATA; //รับ Data จาก from form-interest-product-export.js
let dataInteger = [];
let productInteger = [];
let chart;

let keys = Object.keys(data);
let labelProduct = ['อุปกรณ์อิเล็กทรอนิกส์', 'สุขภาพและความงาม', 'แฟชั่น', 'ของใช้ในบ้านและเครื่องมือต่างๆ', 'เกมและซอฟแวร์'];
let n_labelOutput = labelProduct.length;

// async function shuffleIndex (keys) {
//     let shuffleIndex = 0,
//     valueTemporary;
//     indexKeys = keys.length
//     while (indexKeys--) {
//         shuffleIndex = Math.floor(Math.random() * (indexKeys + 1));
//         valueTemporary = keys[indexKeys];
//         keys[indexKeys] = keys[shuffleIndex];
//         keys[shuffleIndex] = valueTemporary;
//     }
//     return keys
// }

// let newkeys = shuffleIndex(keys);

//แทนข้อมูลให้เป็นตัวเลข
 tf.tidy(() => {
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
            val.product === 'แฟชั่น' ? 2 : val.product === 'ของใช้ในบ้านและเครื่องมือต่างๆ' ? 3 : 4 
        let d = [ageFloat, keySex, keyStatus]       //เก็บข้อมูลที่ แทนด้วยตัวเลข ไว้ใน array d
        dataInteger.push(d)                         //เก็บข้อมูล array d ไว้ใน array dataInteger
        productInteger.push(keyProduct);            //เก็บข้อมูลสิค้าที่แทนด้วยตัวเลข ไว้ไน array productInteger
    }
})





//สร้าง รูปแบบ ข้อมูลเข้า (xs) และ ข้อมูลออก (ys)

//ข้อมูลนำเข้า (input)
let xs = tf.tensor2d(dataInteger);
xs.print();                 //แสดงผล

//ข้อมูลคำตอบ รูปแบบ array1d
let interest1D = tf.tensor1d(productInteger, 'int32');
// interest1D.print();

//ข้อมูลที่ต้องการ (output)
let ys = tf.oneHot(interest1D, n_labelOutput);
ys.print(); 

interest1D.dispose();       //คืนค่า (ไม่จำ)


//สร้าง model NN
let model = tf.sequential();

//สร้าง layers hidden
const hidden = tf.layers.dense({
    units: 25,                  //จำนวนโหนด
    inputShape: [3],            //จำนวนค่าที่รับเข้ามา (input)
    activation: 'relu',         //สูตรสมการที่ใช้
});

//สร้าง layers output
const output = tf.layers.dense({
    units: n_labelOutput,                   //จำนวนโหนด
    activation: 'softmax'       //สูตรสมการที่ใช้
});

//เพิ่ม layers ที่สร้าง ลงใน model
model.add(hidden);
model.add(output);


//กำหนดค่า adam (ค่าละเอียดในการเรียนรู้) ในตัวแปล optimizer
let optimizer = tf.train.adam(0.001);      
model.compile({
    optimizer: optimizer,    //นำตัวแปล optimizer มาใช้

    /*กำหนด loss function ที่ใช้ในการเปรียบเทียบผลลัพธ์ 
    ที่ได้ระหว่างการเรียนรู้ กับ ผลลัพธ์(ที่เรากำหนดให้ ใน ys) */
    loss: 'categoricalCrossentropy'
});

//Train Data
async function train() {
    //เริ่ม Train โดยระบุรายละเอียด ดังนี้
    await model.fit(xs, xs, {   //ข้อมูลสำหรับ Train
        epochs: 200,            //จำนวนรอบ
        callbacks: {            //เพิ่มการทำงานในระหว่าง Train
            //ทำงานเมื่อ Train เสร็จในแต่ระรอบ
            onEpochEnd: (e, l) => {
                //แสดงจำนวนรอบผ่าน console
                console.log(e);    
                //แสดงค่า loss ในรูปแบบทศนิยม 5 ตำแหน่
                console.log(l.loss.toFixed(5));
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

//function รับข้อมูลจาก index.html
function getData() {
    //รับค่ามาจาก index.html ตาม ID
    let getAge = document.getElementById('age').value; 
    let getSex = document.getElementById('sex').value;
    let getStatus = document.getElementById('status').value;

    // แทนข้อมูลให้เป็นตัวเลข เช่นเดียวกับการสร้างข้อมูลให้ AI เรียนรู้
    let getAgeInt = parseInt(getAge);               //แปลง string เป็น int 
    let getKeySex = getSex === 'ชาย' ? 0 : 1 ; 
    let getKeyStatus = getStatus === 'โสด' ? 0 : 1;

    //เพิ่มข้อมูลที่แปลง ลง array arrayData ในรูปแบบ array 2d
    let arrayData = [                               
        [getAgeInt / 100, getKeySex, getKeyStatus]
    ]

    //แปลงข้อมูลที่รับมาในเป็นรูบแปป tensor2d
    let dataTensor2D = tf.tensor2d(arrayData);
    
    //ส่งค่าไปที่ function toAITest เพื่อให้ AI วิเคราะห์
    toAITest(dataTensor2D);                         
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
        let displaylabalProduct = labelProduct[indexLabelProduct];  

        let persen = va * 100 // แปลงจาก ทศนิยม เป็น จำนวนเต็ม
        let displayChartPersen = persen.toFixed(2); //ปรับเป็นทศนิยม 2 ตำแหน่ง   
        
        arrDisplay.push(    //เพิ่ม element (view) เข้าไป array arrDisplay
            `<tr>
                <td>${displaylabalProduct}</td>
                <td>${displayChartPersen}</td>
            </tr>`
        );
        //นำ arrDisplay มาแสดงใน HTML
        document.getElementById('tex').innerHTML = arrDisplay.join('');
    });
}
