mod static_model;

use jni::objects::{AsJArrayRaw, JClass, JString};
use jni::sys::jobjectArray;
use jni::sys::{jint, jlong};
use jni::JNIEnv;

use crate::static_model::StaticModel;

#[no_mangle]
pub extern "C" fn Java_com_ml_shubham0204_model2vec_Model2Vec_create(
    mut env: JNIEnv,
    _class: JClass,
    embeddings_path: JString,
    tokenizer_path: JString,
    num_threads: jint,
) -> jlong {
    let embeddings_path: String = env
        .get_string(&embeddings_path)
        .expect("Could not get embeddings_path")
        .into();
    let tokenizer_path: String = env
        .get_string(&tokenizer_path)
        .expect("Could not get tokenizer_path")
        .into();
    let static_model = StaticModel::new(&embeddings_path, &tokenizer_path, num_threads as usize)
        .expect("Could not instantiate StaticModel");
    Box::into_raw(Box::new(static_model)) as jlong
}

#[no_mangle]
pub extern "C" fn Java_com_ml_shubham0204_model2vec_Model2Vec_addSeqBuffer(
    mut env: JNIEnv,
    _class: JClass,
    model: jlong,
    sequence: JString,
) {
    let model = model as *mut StaticModel;
    let sequence: String = env
        .get_string(&sequence)
        .expect("Could not get sequence")
        .into();
    unsafe {
        (*model).add_seq_buffer(sequence);
    }
}

#[no_mangle]
pub extern "C" fn Java_com_ml_shubham0204_model2vec_Model2Vec_clearSeqBuffer(
    _: JNIEnv,
    _class: JClass,
    model: jlong,
) {
    let model = model as *mut StaticModel;
    unsafe {
        (*model).clear_seq_buffer();
    }
}

#[no_mangle]
pub extern "C" fn Java_com_ml_shubham0204_model2vec_Model2Vec_encode(
    mut env: JNIEnv,
    _class: JClass,
    model: jlong,
) -> jobjectArray {
    let model = model as *mut StaticModel;
    let embeddings = unsafe { (*model).encode_seq_buffer() }.expect("model.encode failed");
    let embedding_dims = unsafe { (*model).embedding_dims };

    let cls = env
        .find_class("[F")
        .expect("Could not find float array class");
    let initial_value = env
        .new_float_array(embedding_dims as i32)
        .expect("Could not create new float array");
    let embeddings_arr = env
        .new_object_array(embeddings.len() as i32, cls, initial_value)
        .expect("Could not create new object array");

    for (i, embedding) in embeddings.iter().enumerate() {
        let embedding_arr = env
            .new_float_array(embedding_dims as i32)
            .expect("Could not create new float array");
        env.set_float_array_region(&embedding_arr, 0, &embedding)
            .expect("Could not set float array region");
        env.set_object_array_element(&embeddings_arr, i as i32, &embedding_arr)
            .expect("Could not set object array element");
        env.delete_local_ref(embedding_arr)
            .expect("Could not delete local reference");
    }

    embeddings_arr.as_jarray_raw()
}

#[no_mangle]
pub extern "C" fn Java_com_ml_shubham0204_model2vec_Model2Vec_release(
    _: JNIEnv,
    _class: JClass,
    model: jlong,
) {
    let model = model as *mut StaticModel;
    unsafe {
        let _ = Box::from_raw(model);
    }
}
