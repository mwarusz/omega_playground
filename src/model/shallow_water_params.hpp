#pragma once

#include <common.hpp>

namespace omega {

struct ShallowWaterParams {
  Real m_f0;
  Real m_grav = 9.81;
  Real m_drag_coeff = 0;
  Real m_visc_del2 = 0;
  Real m_visc_del4 = 0;
  Real m_eddy_diff2 = 0;
  Real m_eddy_diff4 = 0;
  Int m_ntracers = 0;
  bool m_disable_h_tendency = false;
  bool m_disable_vn_tendency = false;
};
} // namespace omega
