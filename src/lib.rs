use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
fn calcular_ventas_por_categoria(categorias: Vec<String>, ventas: Vec<f64>) -> PyResult<HashMap<String, f64>> {
    let mut agrupacion: HashMap<String, f64> = HashMap::new();
    for (cat, v) in categorias.iter().zip(ventas.iter()) {
        *agrupacion.entry(cat.clone()).or_insert(0.0) += v;
    }
    Ok(agrupacion)
}

// EL NOMBRE AQUÍ DEBE SER ventas_app
#[pymodule]
fn ventas_app(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calcular_ventas_por_categoria, m)?)?;
    Ok(())
}