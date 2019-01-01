//เพิ่มข้อมูล
// let con = firebase.database().ref('data-set');
function saveData(){
    let getAge = document.forms["formDataSet"]['age'].value;
    let getSex = document.forms["formDataSet"]['sex'].value;
    let getStatus = document.forms["formDataSet"]['status'].value;
    let getProduct =  document.forms["formDataSet"]['product'].value;
    let data = {
        age: getAge,
        sex: getSex,
        status: getStatus,
        product: getProduct
    }
    con.push(data);
    console.log(data)
}