// Copyright (c) 2013 Spotify AB
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "annoylib.h"
#include "kissrandom.h"
#include "Python.h"
#include "structmember.h"
#include <exception>
#if defined(_MSC_VER) && _MSC_VER == 1500
typedef signed __int32    int32_t;
#else
#include <stdint.h>
#endif


#if defined(ANNOYLIB_USE_AVX512)
#define AVX_INFO "Using 512-bit AVX instructions"
#elif defined(ANNOYLIB_USE_AVX128)
#define AVX_INFO "Using 128-bit AVX instructions"
#else
#define AVX_INFO "Not using AVX instructions"
#endif

#if defined(_MSC_VER)
#define COMPILER_INFO "Compiled using MSC"
#elif defined(__GNUC__)
#define COMPILER_INFO "Compiled on GCC"
#else
#define COMPILER_INFO "Compiled on unknown platform"
#endif

#define ANNOY_DOC (COMPILER_INFO ". " AVX_INFO ".")

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#ifndef Py_TYPE
    #define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
#endif

#ifdef IS_PY3K
    #define PyInt_FromLong PyLong_FromLong 
#endif

using namespace Annoy;

#ifdef ANNOYLIB_MULTITHREADED_BUILD
  typedef AnnoyIndexMultiThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#else
  typedef AnnoyIndexSingleThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#endif

// annoy python object
typedef struct {
  PyObject_HEAD
  int f;
  AnnoyIndexInterface<int32_t, float>* ptr;
} py_annoy;


static PyObject *
py_an_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  py_annoy *self = (py_annoy *)type->tp_alloc(type, 0);
  if (self == NULL) {
    return NULL;
  }
  const char *metric = NULL;

  static char const * kwlist[] = {"f", "metric", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|s", (char**)kwlist, &self->f, &metric))
    return NULL;
  if (!metric) {
    // This keeps coming up, see #368 etc
    PyErr_WarnEx(PyExc_FutureWarning, "The default argument for metric will be removed "
		 "in future version of Annoy. Please pass metric='angular' explicitly.", 1);
    self->ptr = new AnnoyIndex<int32_t, float, Angular, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(self->f);
  } else if (!strcmp(metric, "angular")) {
   self->ptr = new AnnoyIndex<int32_t, float, Angular, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(self->f);
  } else if (!strcmp(metric, "euclidean")) {
    self->ptr = new AnnoyIndex<int32_t, float, Euclidean, Kiss64Random, AnnoyIndexThreadedBuildPolicy>(self->f);
  } else {
    PyErr_SetString(PyExc_ValueError, "No such metric");
    return NULL;
  }

  return (PyObject *)self;
}


static int 
py_an_init(py_annoy *self, PyObject *args, PyObject *kwargs) {
  // Seems to be needed for Python 3
  const char *metric = NULL;
  int f;
  static char const * kwlist[] = {"f", "metric", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|s", (char**)kwlist, &f, &metric))
    return (int) NULL;
  return 0;
}


static void 
py_an_dealloc(py_annoy* self) {
  delete self->ptr;
  Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyMemberDef py_annoy_members[] = {
  {(char*)"f", T_INT, offsetof(py_annoy, f), 0,
   (char*)""},
  {NULL}	/* Sentinel */
};


static PyObject *
py_an_load(py_annoy *self, PyObject *args, PyObject *kwargs) {
  char *filename, *error;
  bool prefault = false;
  if (!self->ptr) 
    return NULL;
  static char const * kwlist[] = {"fn", "prefault", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|b", (char**)kwlist, &filename, &prefault))
    return NULL;

  if (!self->ptr->load(filename, prefault, &error)) {
    PyErr_SetString(PyExc_IOError, error);
    free(error);
    return NULL;
  }
  Py_RETURN_TRUE;
}


static PyObject *
py_an_save(py_annoy *self, PyObject *args, PyObject *kwargs) {
  char *filename, *error;
  bool prefault = false;
  if (!self->ptr) 
    return NULL;
  static char const * kwlist[] = {"fn", "prefault", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|b", (char**)kwlist, &filename, &prefault))
    return NULL;

  if (!self->ptr->save(filename, prefault, &error)) {
    PyErr_SetString(PyExc_IOError, error);
    free(error);
    return NULL;
  }
  Py_RETURN_TRUE;
}


PyObject*
get_nns_to_python(const vector<int32_t>& result, const vector<float>& distances, int include_distances) {
  PyObject* l = NULL;
  PyObject* d = NULL;
  PyObject* t = NULL;

  if ((l = PyList_New(result.size())) == NULL) {
    goto error;
  }
  for (size_t i = 0; i < result.size(); i++) {
    PyObject* res = PyInt_FromLong(result[i]);
    if (res == NULL) {
      goto error;
    }
    PyList_SetItem(l, i, res);
  }
  if (!include_distances)
    return l;

  if ((d = PyList_New(distances.size())) == NULL) {
    goto error;
  }

  for (size_t i = 0; i < distances.size(); i++) {
    PyObject* dist = PyFloat_FromDouble(distances[i]);
    if (dist == NULL) {
      goto error;
    }
    PyList_SetItem(d, i, dist);
  }

  if ((t = PyTuple_Pack(2, l, d)) == NULL) {
    goto error;
  }

  return t;

  error:
    Py_XDECREF(l);
    Py_XDECREF(d);
    Py_XDECREF(t);
    return NULL;
}


bool check_constraints(py_annoy *self, int32_t item, bool building) {
  if (item < 0) {
    PyErr_SetString(PyExc_IndexError, "Item index can not be negative");
    return false;
  } else if (!building && item >= self->ptr->get_n_items()) {
    PyErr_SetString(PyExc_IndexError, "Item index larger than the largest item index");
    return false;
  } else {
    return true;
  }
}

static PyObject* 
py_an_get_nns_by_item(py_annoy *self, PyObject *args, PyObject *kwargs) {
  int32_t item, n, search_k=-1, include_distances=0;
  if (!self->ptr) 
    return NULL;

  static char const * kwlist[] = {"i", "n", "search_k", "include_distances", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|ifi", (char**)kwlist, &item, &n, &search_k, &include_distances))
    return NULL;

  if (!check_constraints(self, item, false)) {
    return NULL;
  }

  vector<int32_t> result;
  vector<float> distances;

  Py_BEGIN_ALLOW_THREADS;
  self->ptr->get_nns_by_item(item, n, search_k, &result, include_distances ? &distances : NULL);
  Py_END_ALLOW_THREADS;

  return get_nns_to_python(result, distances, include_distances);
}


bool
convert_list_to_vector(PyObject* v, int f, vector<float>* w) {
  Py_ssize_t length = PyObject_Size(v);
  if (length == -1) {
    return false;
  }
  if (length != f) {
    PyErr_Format(PyExc_IndexError, "Vector has wrong length (expected %d, got %ld)", f, length);
    return false;
  }

  for (int z = 0; z < f; z++) {
    PyObject *key = PyInt_FromLong(z);
    if (key == NULL) {
      return false;
    }
    PyObject *pf = PyObject_GetItem(v, key);
    Py_DECREF(key);
    if (pf == NULL) {
      return false;
    }
    double value = PyFloat_AsDouble(pf);
    Py_DECREF(pf);
    if (value == -1.0 && PyErr_Occurred()) {
      return false;
    }
    (*w)[z] = value;
  }
  return true;
}

static PyObject* 
py_an_get_nns_by_vector(py_annoy *self, PyObject *args, PyObject *kwargs) {
  PyObject* v;
  int32_t n, search_k=-1, include_distances=0;
  if (!self->ptr) 
    return NULL;

  static char const * kwlist[] = {"vector", "n", "search_k", "include_distances", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ifi", (char**)kwlist, &v, &n, &search_k, &include_distances))
    return NULL;

  vector<float> w(self->f);
  if (!convert_list_to_vector(v, self->f, &w)) {
    return NULL;
  }

  vector<int32_t> result;
  vector<float> distances;

  Py_BEGIN_ALLOW_THREADS;
  self->ptr->get_nns_by_vector(&w[0], n, search_k, &result, include_distances ? &distances : NULL);
  Py_END_ALLOW_THREADS;

  return get_nns_to_python(result, distances, include_distances);
}


static PyObject* 
py_an_get_item_vector(py_annoy *self, PyObject *args) {
  int32_t item;
  if (!self->ptr) 
    return NULL;
  if (!PyArg_ParseTuple(args, "i", &item))
    return NULL;

  if (!check_constraints(self, item, false)) {
    return NULL;
  }

  vector<float> v(self->f);
  self->ptr->get_item(item, &v[0]);
  PyObject* l = PyList_New(self->f);
  if (l == NULL) {
    return NULL;
  }
  for (int z = 0; z < self->f; z++) {
    PyObject* dist = PyFloat_FromDouble(v[z]);
    if (dist == NULL) {
      goto error;
    }
    PyList_SetItem(l, z, dist);
  }

  return l;

  error:
    Py_XDECREF(l);
    return NULL;
}


static PyObject* 
py_an_add_item(py_annoy *self, PyObject *args, PyObject* kwargs) {
  PyObject* v;
  int32_t item;
  float weight = 1;
  if (!self->ptr) 
    return NULL;
  static char const * kwlist[] = {"i", "vector", "w", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO|f", (char**)kwlist, &item, &v, &weight))
    return NULL;

  if (!check_constraints(self, item, true)) {
    return NULL;
  }

  vector<float> w(self->f);
  if (!convert_list_to_vector(v, self->f, &w)) {
    return NULL;
  }

  char* error;
  if (!self->ptr->add_item(item, &w[0], weight, &error)) {
    PyErr_SetString(PyExc_Exception, error);
    free(error);
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject *
py_an_on_disk_build(py_annoy *self, PyObject *args, PyObject *kwargs) {
  char *filename, *error;
  if (!self->ptr)
    return NULL;
  static char const * kwlist[] = {"fn", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &filename))
    return NULL;

  if (!self->ptr->on_disk_build(filename, &error)) {
    PyErr_SetString(PyExc_IOError, error);
    free(error);
    return NULL;
  }
  Py_RETURN_TRUE;
}

static PyObject *
py_an_build(py_annoy *self, PyObject *args, PyObject *kwargs) {
  int q, n_neighbors=0, n_jobs = -1;
  float top_p = 1;
  if (!self->ptr) 
    return NULL;
  static char const * kwlist[] = {"n_trees", "top_p", "n_neighbors", "n_jobs", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|fii", (char**)kwlist, &q, &top_p, &n_neighbors, &n_jobs))
    return NULL;

  bool res;
  char* error;
  Py_BEGIN_ALLOW_THREADS;
  res = self->ptr->build(q, top_p, n_neighbors, n_jobs, &error);
  Py_END_ALLOW_THREADS;
  if (!res) {
    PyErr_SetString(PyExc_Exception, error);
    free(error);
    return NULL;
  }

  Py_RETURN_TRUE;
}


static PyObject *
py_an_unbuild(py_annoy *self) {
  if (!self->ptr) 
    return NULL;

  char* error;
  if (!self->ptr->unbuild(&error)) {
    PyErr_SetString(PyExc_Exception, error);
    free(error);
    return NULL;
  }

  Py_RETURN_TRUE;
}


static PyObject *
py_an_unload(py_annoy *self) {
  if (!self->ptr) 
    return NULL;

  self->ptr->unload();

  Py_RETURN_TRUE;
}


static PyObject *
py_an_get_distance(py_annoy *self, PyObject *args) {
  int32_t i, j;
  if (!self->ptr) 
    return NULL;
  if (!PyArg_ParseTuple(args, "ii", &i, &j))
    return NULL;

  if (!check_constraints(self, i, false) || !check_constraints(self, j, false)) {
    return NULL;
  }

  double d = self->ptr->get_distance(i,j);
  return PyFloat_FromDouble(d);
}


static PyObject *
py_an_get_n_items(py_annoy *self) {
  if (!self->ptr) 
    return NULL;

  int32_t n = self->ptr->get_n_items();
  return PyInt_FromLong(n);
}

static PyObject *
py_an_get_n_trees(py_annoy *self) {
  if (!self->ptr) 
    return NULL;

  int32_t n = self->ptr->get_n_trees();
  return PyInt_FromLong(n);
}

static PyObject *
py_an_verbose(py_annoy *self, PyObject *args) {
  int verbose;
  if (!self->ptr) 
    return NULL;
  if (!PyArg_ParseTuple(args, "i", &verbose))
    return NULL;

  self->ptr->verbose((bool)verbose);

  Py_RETURN_TRUE;
}


static PyObject *
py_an_set_seed(py_annoy *self, PyObject *args) {
  int q;
  if (!self->ptr)
    return NULL;
  if (!PyArg_ParseTuple(args, "i", &q))
    return NULL;

  self->ptr->set_seed(q);

  Py_RETURN_NONE;
}

static PyObject *
py_an_set_model(py_annoy *self, PyObject *args, PyObject *kwargs) {
  PyObject* coef;
  float intercept = 0;
  if (!self->ptr) 
    return NULL;
  static char const * kwlist[] = {"coef", "intercept", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Of", (char**)kwlist, &coef, &intercept))
    return NULL;
  
  vector<float> w(self->f);
  if (!convert_list_to_vector(coef, self->f, &w)) {
    return NULL;
  }

  self->ptr->set_model(&w[0], intercept);

  Py_RETURN_NONE;
}


static PyMethodDef AnnoyMethods[] = {
  {"load",	(PyCFunction)py_an_load, METH_VARARGS | METH_KEYWORDS, "Loads (mmaps) an index from disk."},
  {"save",	(PyCFunction)py_an_save, METH_VARARGS | METH_KEYWORDS, "Saves the index to disk."},
  {"get_nns_by_item",(PyCFunction)py_an_get_nns_by_item, METH_VARARGS | METH_KEYWORDS, "Returns the `n` closest items to item `i`.\n\n:param search_k: the query will inspect up to `search_k` nodes.\n`search_k` gives you a run-time tradeoff between better accuracy and speed.\n`search_k` defaults to `n_trees * n` if not provided.\n\n:param include_distances: If `True`, this function will return a\n2 element tuple of lists. The first list contains the `n` closest items.\nThe second list contains the corresponding distances."},
  {"get_nns_by_vector",(PyCFunction)py_an_get_nns_by_vector, METH_VARARGS | METH_KEYWORDS, "Returns the `n` closest items to vector `vector`.\n\n:param search_k: the query will inspect up to `search_k` nodes.\n`search_k` gives you a run-time tradeoff between better accuracy and speed.\n`search_k` defaults to `n_trees * n` if not provided.\n\n:param include_distances: If `True`, this function will return a\n2 element tuple of lists. The first list contains the `n` closest items.\nThe second list contains the corresponding distances."},
  {"get_item_vector",(PyCFunction)py_an_get_item_vector, METH_VARARGS, "Returns the vector for item `i` that was previously added."},
  {"add_item",(PyCFunction)py_an_add_item, METH_VARARGS | METH_KEYWORDS, "Adds item `i` (any nonnegative integer) with vector `v`.\n\nNote that it will allocate memory for `max(i)+1` items."},
  {"on_disk_build",(PyCFunction)py_an_on_disk_build, METH_VARARGS | METH_KEYWORDS, "Build will be performed with storage on disk instead of RAM."},
  {"build",(PyCFunction)py_an_build, METH_VARARGS | METH_KEYWORDS, "Builds a forest of `n_trees` trees.\n\nMore trees give higher precision when querying. After calling `build`,\nno more items can be added. `n_jobs` specifies the number of threads used to build the trees. `n_jobs=-1` uses all available CPU cores."},
  {"unbuild",(PyCFunction)py_an_unbuild, METH_NOARGS, "Unbuilds the tree in order to allows adding new items.\n\nbuild() has to be called again afterwards in order to\nrun queries."},
  {"unload",(PyCFunction)py_an_unload, METH_NOARGS, "Unloads an index from disk."},
  {"get_distance",(PyCFunction)py_an_get_distance, METH_VARARGS, "Returns the distance between items `i` and `j`."},
  {"get_n_items",(PyCFunction)py_an_get_n_items, METH_NOARGS, "Returns the number of items in the index."},
  {"get_n_trees",(PyCFunction)py_an_get_n_trees, METH_NOARGS, "Returns the number of trees in the index."},
  {"verbose",(PyCFunction)py_an_verbose, METH_VARARGS, ""},
  {"set_seed",(PyCFunction)py_an_set_seed, METH_VARARGS, "Sets the seed of Annoy's random number generator."},
  {"set_model",(PyCFunction)py_an_set_model, METH_VARARGS | METH_KEYWORDS, "Sets the model."},
  {NULL, NULL, 0, NULL}		 /* Sentinel */
};


static PyTypeObject PyAnnoyType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "annoy.Annoy",          /*tp_name*/
  sizeof(py_annoy),       /*tp_basicsize*/
  0,                      /*tp_itemsize*/
  (destructor)py_an_dealloc, /*tp_dealloc*/
  0,                      /*tp_print*/
  0,                      /*tp_getattr*/
  0,                      /*tp_setattr*/
  0,                      /*tp_compare*/
  0,                      /*tp_repr*/
  0,                      /*tp_as_number*/
  0,                      /*tp_as_sequence*/
  0,                      /*tp_as_mapping*/
  0,                      /*tp_hash */
  0,                      /*tp_call*/
  0,                      /*tp_str*/
  0,                      /*tp_getattro*/
  0,                      /*tp_setattro*/
  0,                      /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  ANNOY_DOC,              /* tp_doc */
  0,                      /* tp_traverse */
  0,                      /* tp_clear */
  0,                      /* tp_richcompare */
  0,                      /* tp_weaklistoffset */
  0,                      /* tp_iter */
  0,                      /* tp_iternext */
  AnnoyMethods,           /* tp_methods */
  py_annoy_members,       /* tp_members */
  0,                      /* tp_getset */
  0,                      /* tp_base */
  0,                      /* tp_dict */
  0,                      /* tp_descr_get */
  0,                      /* tp_descr_set */
  0,                      /* tp_dictoffset */
  (initproc)py_an_init,   /* tp_init */
  0,                      /* tp_alloc */
  py_an_new,              /* tp_new */
};

static PyMethodDef module_methods[] = {
  {NULL}	/* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "annoylib",          /* m_name */
    ANNOY_DOC,           /* m_doc */
    -1,                  /* m_size */
    module_methods,      /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#endif

PyObject *create_module(void) {
  PyObject *m;

  if (PyType_Ready(&PyAnnoyType) < 0)
    return NULL;

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&moduledef);
#else
  m = Py_InitModule("annoylib", module_methods);
#endif

  if (m == NULL)
    return NULL;

  Py_INCREF(&PyAnnoyType);
  PyModule_AddObject(m, "Annoy", (PyObject *)&PyAnnoyType);
  return m;
}

#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_annoylib(void) {
    return create_module();      // it should return moudule object in py3
  }
#else
  PyMODINIT_FUNC initannoylib(void) {
    create_module();
  }
#endif


// vim: tabstop=2 shiftwidth=2
