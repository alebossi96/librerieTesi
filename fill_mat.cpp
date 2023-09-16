#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

void fill_mat(std::vector<int>& dims, PyObject* data, PyObject* col, PyObject* output_col, PyArrayObject* mat, int idx) {
    int dims_count = dims.size();

    if (dims_count == 0) {
        for (int i = 0; i < PyList_Size(output_col); i++) {
            PyObject* el = PyList_GetItem(output_col, i);
            PyObject* value = PyObject_CallMethod(data, "get", "O", el);
            double result = PyFloat_AsDouble(PyList_GetItem(value, 0));
            *(double*)PyArray_GETPTR1(mat, idx) = result;
            Py_DECREF(value);
        }
    } else {
        PyObject* col0 = PyList_GetItem(col, 0);
        PyObject* el_columns = PyObject_CallMethod(data, "unique", "O", col0);
        int len_el_columns = PyList_Size(el_columns);

        for (int i = 0; i < len_el_columns; i++) {
            PyObject* el = PyList_GetItem(el_columns, i);
            PyObject* selected = PyObject_CallMethod(data, "filter", "OO", col0, el);

            fill_mat(dims, selected, col, output_col, mat, i);
            Py_DECREF(selected);
        }
        Py_DECREF(el_columns);
    }
}

static PyObject* wrap_fill_mat(PyObject* self, PyObject* args) {
    PyObject* data;
    PyObject* col;
    PyObject* output_col;
    PyObject* dims_obj;

    if (!PyArg_ParseTuple(args, "OOOO", &dims_obj, &data, &col, &output_col)) {
        return NULL;
    }

    std::vector<int> dims;
    int len_dims_obj = PyList_Size(dims_obj);
    for (int i = 0; i < len_dims_obj; i++) {
        dims.push_back(PyLong_AsLong(PyList_GetItem(dims_obj, i)));
    }

    npy_intp np_dims[len_dims_obj + 1];
    std::copy(dims.begin(), dims.end(), np_dims);
    np_dims[len_dims_obj] = PyList_Size(output_col);

    PyArrayObject* mat = (PyArrayObject*)PyArray_SimpleNew(len_dims_obj + 1, np_dims, NPY_DOUBLE);
    fill_mat(dims, data, col, output_col, mat, 0);

    return (PyObject*)mat;
}

static PyMethodDef FillMatMethods[] = {
    {"fill_mat", wrap_fill_mat, METH_VARARGS, "Fill mat"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fillmatmodule = {
    PyModuleDef_HEAD_INIT,
    "fill_mat_module",
    NULL,
    -1,
    FillMatMethods
};

PyMODINIT_FUNC PyInit_fill_mat_module(void) {
    PyObject* module = PyModule_Create(&fillmatmodule);
    if (module == NULL)
        return NULL;
    import_array();
    if (PyErr_Occurred()) return NULL;
    return module;
}

