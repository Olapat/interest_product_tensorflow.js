// let aVGA = answersVGA_Card;
// let aram = answersRam;
// let aAll = answers;
// let testInput = input_3D;
// let testaAll_2d = [];
// let testaAll_1d = [];

// let qProduct = question_product;
// let qDeProduct = question_detail_product;

// let model;
// let qIndexP = [];
// let qIndexPD = [];
// let qIndexOneHotP = [];
// let qIndexOneHotPD = [];
// let indexResulte = [];


// for (let i = 0; i < qProduct.length; i++) {
//     let y = i / qProduct.length;
//     qIndexP.push([y]);
//     qIndexOneHotP.push(i);
// }

// for (let i = 0; i < qDeProduct.length; i++) {
//     let y = i / qDeProduct.length;
//     qIndexPD.push([y]);
//     qIndexOneHotPD.push(i);
// }

// for (let i = 0; i < aAll.length; i++) {
//     let y = i / aAll.length;
//     testaAll_2d.push([y]);
//     testaAll_1d.push(i);
// }

// let xs = tf.tensor2d(qIndexP);
// xs.print();
// let indexTensorP = tf.tensor1d(qIndexOneHotP, 'int32');
// indexTensorP.print();
// indexTensorP.dispose();


// let ys = tf.oneHot(indexTensorP, qProduct.length).cast('float32');
// ys.print();


// console.log(testaAll_1d);
// let testInput_1done = tf.tensor1d(testaAll_1d, 'int32');
// testInput_1done.print();
// let ysAllasw = tf.oneHot(testInput_1done, testaAll_1d.length).cast('float32');
// ysAllasw.print();

// let qet = tf.tensor3d(aAll);
// qet.print();

// let indexTensorPD = tf.tensor1d(qIndexOneHotPD, 'int32');
// let ys2 = tf.oneHot(indexTensorPD, qDeProduct.length).cast('float32');
// // ys2.print();

// model = tf.sequential();
// const hidden = tf.layers.dense({ //เข้า Product
//     units: 4,
//     inputShape: [1],
//     activation: 'sigmoid',
//     useBias: true               //ใช้ค่าเบี่ยงเบน เพื่อช่วยในการคำนวณ

// });

// const output = tf.layers.dense({
//     units: 4,
//     activation: 'softmax'
// });

// // const hidden2 = tf.layers.dense({ //เข้า คำถาม
// //     units: 4,
// //     inputShape: [4],
// //     activation: 'sigmoid'  // || 'relu'
// // });

// model.add(hidden);
// // model.add(hidden2);
// model.add(output);


// const optimizer = tf.train.sgd(1.0); // || adam();

// model.compile({
//     optimizer: optimizer,
//     loss: 'categoricalCrossentropy',
//     matrics: ['accuracy'] //กำหนด ค่า accuracy (ค่าความแม่นยำ)
// });

// function draw() {
//     let rr = []
//     tf.tidy(() => {
//         const input = tf.tensor2d(qIndexP);
//         let resulte = model.predict(input);
//         resulte.print();
//         let argMax = resulte.argMax(1);
//         // argMax.print();
        
//         for (let i = 0; i < qProduct.length; i++) {
//             let ll = argMax.dataSync()[i];
//             rr.push(ll);
//         }
//         // console.log(rr);
        
//     });
//     return rr;
// };

// // train();

// async function train () {
//     await model.fit(xs, ys, {
        // shuffle: true,                              //สลับข้อมูล 
        // validationSplit: 0.01,                  //กำหนดค่าความสูญเสียของข้อมูล
        // batchSize: s.length,
//         epochs: 1,
//         callbacks: {
//             onEpochEnd: (epoch, logs) => {
//                 console.log(epoch);
//                 console.log(logs.loss);
//             },
//             // onBatchEnd: async () => {
//             //     await tf.nextFrame();
//             // },
//             onTrainEnd: () => {
//                 console.log("เสร็จ");
//             }
//         }
//     });

//    indexResulte = draw();
// }

// function getinput() {
//     let input = document.getElementById('input').value;
//     let i= 0;
//     let d = -1;
//     let t, tt;

//     while (d === -1) {
//         d = input.search(qProduct[i])
//         t = i;
//         i++;
//     }
//     if (t >= qProduct.length) {
//         console.log("ไม่เจอ");
//     }else {
//         detail(input, t);
//         // model2(t,tt);
//     }
// };

// let td = 0;
// function detail(input, t) {
//     let d = -1;
//     let i= 0;
//     let te;
    
//     while (d === -1) {
//         d = input.search(qDeProduct[t][td][i])
//         if (i === qDeProduct[t][td].length) {
//             td++;
//         }
//         te = i;
//         i++;
//         console.log("td ", td)
//     }
//     if (te >= qDeProduct.length && d === -1) {
//         console.log("ไม่เจอP");
//     }else {
//         // model2(te);
//         console.log("เจอแล้ว ", t , " " , td, );
//         console.log(qDeProduct[t][td])
//         console.log(aAll[t][td])
//         td = 0;
//     }
// }

// function model2(indexP, indexPD) {
//     // console.log(indexResulte);
//     function findFirstLargeNumber(element) {
//         return element === indexP;
//       }
//     let indexSearchP  = qProduct.findIndex(findFirstLargeNumber);
//     console.log("เจอ " , indexSearchP);
//     let indexSearchPD = qDeProduct.findIndex(findFirstLargeNumber);
// }
