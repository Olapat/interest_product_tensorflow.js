const data = DATA;
let inputData = [];
let outputData = [];
let labalProduct;

//เตรียมข้อมูล แปลงเป็นตัวเลข / แยก Input Output
const keys = Object.keys(data);         //ดึงเอา key จากข้อมูล
const numData = keys.length;            //จำนวนข้อมูลทั้งหมด
tf.util.shuffle(keys);                  //สลับข้อมูล
console.log(keys);

tf.tidy(()=> {            //tidy เป็น function ที่จะคืนค่า memory หลังจากทำงานเสร็จ
    //วนรูปเพื่อนำข้อมูลแต่ละตัวมาแปลงเป็นตัวเลข และแยกออกเป็น Input, Output
    for (const key of keys) {    //วนรูปตามจำนวนข้อมูล
        let value = data[key];   //เก็บข้อมูลของ key นั้นๆ ไว้ในตัวแปล value
        let ageInt = parseInt(value.age); //แปลงข้อมูลอายุจาก string เป็น Integer
        let ageFloat = ageInt / 100;  //หารด้วย 100 เพื่อลดค่าให้เหลือระหว่าง 0 - 1

        //ตรวจสอบ เพศว่าเท่ากับ ชาย หรือไม่ ถ้าใช้ ให้แทนเป็นตัวเลข  0 ถ้าไม่ แทนเป็น 1
        let keySex = value.sex === 'ชาย' ? 0 : 1;     
        
        //ตรวจสอบ สถานะว่าเท่ากับ โสด หรือไม่ ถ้าใช้ ให้แทนเป็นตัวเลข  0 ถ้าไม่ แทนเป็น 1
        let keyStatus = value.status === 'โสด' ? 0 : 1;

        //เพิ่มข้อมูลที่แปลงแล้วลงใน array inputData เป็นข้อมูล Input
        inputData.push([ageFloat, keySex, keyStatus]);  

        //เก็บค่าจาก productRate ไว้ในตัวแปล valueAllProductRate
        let valueAllProductRate = Object.values(data[key].productRate);

        let valueProductRate = [];
        //วนรูปเพื่อนำเอาค่าคะแนนจาก productRate (คะแนน) มาแปลงเป็นเป็นตัวเลข
        valueAllProductRate.forEach((va) => {
            /*แปลงข้อมูลคะแนนจาก string เป็น Integer แล้ว 
            หาร 5 (คะแนนสูงสุด) เพื่อลดค่าให้เหลือระหว่าง 0 - 1 */
            let valueFloat = parseInt(va) / 5;       

            //ปรับให้เป็น ทศนิยม 1 ตำแหน่ง
            let valueStringFloatOne = valueFloat.toFixed(1);        

            /*เนื่องจากการปรับทศนิยมการบรรทัดที่แล้วด้วยคำสั้ง toFixed 
            ทำให้ค่ากลับเป็น string ดังนั้นจึงต้องแปลงค่าให้เป็น float */
            let valueFloatOne = parseFloat(valueStringFloatOne);

            //เพิ่มข้อมูลที่แปลงแล้วลงใน array valueProductRate
            valueProductRate.push(valueFloatOne);                   
        });

        //เพิ่ม valueProductRate ลงใน array outputData
        outputData.push(valueProductRate);             
        
        //เก็บคีย์จาก productRate ไว้ในตัวแปล labalProduct
        labalProduct = Object.keys(data[key].productRate);  
    }
}); 

// console.log(dataInteger);
// console.log(rateProduct);
// console.log(labalProduct);

const numTestData = Math.round(numData * 0.8);      //กำหนดขนาดของข้อมูลไว้สำหรับ test 80%
const numTrainData = numData - numTestData;         //กำหนดขนาดของข้อมูลที่เหลือไว้สำหรับ train
const xDim = inputData[0].length;                   //กำหนดขนาด Input (3)
const numClass = labalProduct.length;               //จำนวนข้อมูล Class (คำตอบ)

//สร้าง รูปแบบ ข้อมูลเข้า (xs) และ ข้อมูลออก (ys)

//ข้อมูลนำเข้า (input)

let xs = tf.tensor2d(inputData);                    //สร้าง Input เป็น tensor2d
xs.print();                 //แสดงผล

let ys = tf.tensor2d(outputData);                   //สร้าง Output เป็น tensor2d
ys.print();

//แบ่งข้อมูลไว้สำหรับ Train และ Test
const xTrain = xs.slice([0, 0], [numTrainData, xDim]);          //แบ่งข้อมูล Input สำหรับ Train
const xTest = xs.slice([numTrainData, 0], [numTestData, xDim]); //แบ่งข้อมูล Input สำหรับ Test
const yTrain = ys.slice([0, 0], [numTrainData, numClass]);      //แบ่งข้อมูล Output สำหรับ Train
const yTest = ys.slice([0, 0], [numTestData, numClass]);        //แบ่งข้อมูล Output สำหรับ Test

//สร้าง model NN
let model = tf.sequential();        //สร้าง model

//สร้าง layers hidden
const hidden = tf.layers.dense({
    units: 50,                      //จำนวนโหนด
    inputShape: [xDim],             //จำนวนค่าที่รับเข้ามา (input)
    activation: 'sigmoid',          //function ที่ใช้ในการคำนวณ
});

//สร้าง layers output
const output = tf.layers.dense({
    units: numClass,                //จำนวนโหนด
    activation: 'softmax',          //function ที่ใช้ในการคำนวณ
});

//เพิ่ม layers ที่สร้าง ลงใน model
model.add(hidden);
model.add(output);

let optimizer = tf.train.adam(0.02);    //กำหนดค่า adam (ค่าละเอียดในการเรียนรู้) ในตัวแปล optimizer
model.compile({
    optimizer: optimizer,               //นำตัวแปล optimizer มาใช้
    loss: 'categoricalCrossentropy',    //กำหนด loss function ที่ใช้ในการเปรียบเทียบผลลัพธ์ ที่ได้ระหว่างการเรียนรู้ กับ ผลลัพธ์(ที่เรากำหนดให้ ใน ys)
    metrics: ['accuracy']               //กำหนด ค่า accuracy (ค่าความแม่นยำ)
});

//Train Data
async function train() {
    //เริ่ม Train โดยระบุรายละเอียด ดังนี้
    await model.fit(xTrain, yTrain, {               //ข้อมูลสำหรับ Train
        epochs: 500,                                //จำนวนรอบ
        validationData: [xTest, yTest],             //ข้อมูลสำหรับ Test
        callbacks: {                                //เพิ่มการทำงานในระหว่าง Train
            //ทำงานเมื่อ Train เสร็จในแต่ระรอบ
            onEpochEnd: (e, l) => {                 
                console.log(e);                                 //แสดงจำนวนรอบผ่าน console
                console.log(l.loss.toFixed(5), ':', l.acc);     //แสดงค่า loss ในรูปแบบทศนิยม 5 ตำแหน่ง และ ค่า acc ผ่าน console 
            },
            //ทำงานเมื่อ Train ครบแล้ว
            onTrainEnd: () => {                     
                console.log("เทรดเสร็จแล้ว");          //แสดงข้อความเตือนผ่าน console
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
    tf.tidy(() => {                                 //tidy เป็น function ที่จะคืนค่า memory หลังจากทำงานเสร็จ
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
    displayResults(chartMaxToMin);      
}

// function แสดงผลลัพธ์ บน html ตามลำดับ
function displayResults(resultsChart) {
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
