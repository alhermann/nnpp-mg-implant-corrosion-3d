//
// space.h
//
// Simple vector and matrix classes
//

#ifndef __SPACE_H__

#define __SPACE_H__

#include <exception>

namespace space {
template <typename T, typename I = size_t>
class vector {
 public:
  vector() : _data(0), _size(0) {
    // Nothing to do!
  }

  vector(I _size) : _size(_size) { _data = new T[_size]; }

  ~vector() {
    if (_data != 0) {
      delete[] _data;
    }
  }

  void clear() {
    for (int i = 0; i < _size; i++) {
      _data[i] = T();
    }
  }

  void swap(vector<T, I> &_v) {
    T *_v_data = _v._data;
    _v._data = _data;
    _data = _v_data;
  }

  void swap(Eigen::VectorXd &_v) {
    for (int i = 0; i < _v.size(); ++i) _data[i] = _v[i];
  }

  T &operator[](const I _i) const {
#ifdef _DEBUG
    if ((_i < 0) || (_i >= _size)) {
      throw std::exception("space::vector::operator[]: index out of range");
    }
#endif

    return _data[_i];
  }

  void resize(I _new_size) {
    if (_data != 0) {
      delete[] _data;
    }

    _size = _new_size;
    _data = new T[_new_size];
  }

  I size() const { return _size; }

  T *data() const { return _data; }

  int get_index(const T val) const {
    int idx = std::distance(_data, std::find(_data, _data + _size, val));
    if (idx != _size)
      return idx;
    else
      return -1;
  }

  Eigen::VectorXd return_eigen(Eigen::VectorXd &vec) {
    for (int i = 0; i < _size; i++) {
      vec[i] = _data[i];
    }
    return vec;
  }

 private:
  T *_data;
  I _size;
};

template <typename T, typename I = size_t>
class matrix {
 public:
  matrix() : _data(0), _size1(0), _size2(0) {
    // Nothing to do!
  }

  matrix(I _size1, I _size2) : _size1(_size1), _size2(_size2) {
    _data = new T[_size1 * _size2];
  }

  ~matrix() {
    if (_data != 0) {
      delete[] _data;
    }
  }

  void clear() {
    for (int i = 0; i < _size1 * _size2; i++) {
      _data[i] = T();
    }
  }

  void swap(matrix<T, I> &_m) {
    T *_m_data = _m._data;
    _m._data = _data;
    _data = _m_data;
  }

  T &operator()(const I _i, const I _j) const {
#ifdef _DEBUG
    if ((_i < 0) || (_j < 0) || (_i >= _size1) || (_j >= _size2)) {
      throw std::exception("space::vector::operator(): index out of range");
    }
#endif

    return _data[_i * _size2 + _j];
  }

  void resize(I _new_size1, I _new_size2) {
    if (_data != 0) {
      delete[] _data;
    }

    _size1 = _new_size1;
    _size2 = _new_size2;
    _data = new T[_new_size1 * _new_size2];
  }

  I rows() const { return _size1; }

  I cols() const { return _size2; }

  T *data() const { return _data; }

 private:
  T *_data;
  I _size1;
  I _size2;
};
}  // namespace space

#endif  // __SPACE_H__
