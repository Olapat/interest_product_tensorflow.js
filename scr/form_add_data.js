//เพิ่มข้อมูล
let con = firebase.database().ref('data-set');
function saveData(){
    let getAge = document.forms["formDataSet"]['age'].value;
    let getSex = document.forms["formDataSet"]['sex'].value;
    let getStatus = document.forms["formDataSet"]['status'].value;
    let getProductE =  document.forms["formDataSet"]['อุปกรณ์อิเล็กทรอนิกส์'].value;
    let getProductS =  document.forms["formDataSet"]['สุขภาพและความงาม'].value;
    let getProductF =  document.forms["formDataSet"]['แฟชั่น'].value;
    let getProductH =  document.forms["formDataSet"]['ของใช้ในบ้านและเครื่องมือต่างๆ'].value;
    let getProductG =  document.forms["formDataSet"]['เกมและซอฟแวร์'].value;
    let data = {
        age: getAge,
        sex: getSex,
        status: getStatus,
        productRate: {
            'อุปกรณ์อิเล็กทรอนิกส์': getProductE,
            'สุขภาพและความงาม': getProductS,
            'แฟชั่น': getProductF,
            'ของใช้ในบ้านและเครื่องมือต่างๆ': getProductH,
            'เกมและซอฟแวร์': getProductG
        }
    }
    con.push(data);
    console.log(data)
}