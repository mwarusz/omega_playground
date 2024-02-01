#pragma once

#include <common.hpp>
#include <planar_hexagonal_mesh.hpp>

namespace omega {

enum class AddMode { replace, increment };

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

struct LinearShallowWaterParams : ShallowWaterParams {
  Real m_h0;
};

// fwd
struct ShallowWaterModelBase;

struct ShallowWaterState {
  Real2d m_h_cell;
  Real2d m_vn_edge;
  Real3d m_tr_cell;

  ShallowWaterState(MPASMesh *mesh, Int ntracers);
  ShallowWaterState(const ShallowWaterModelBase &sw);
};

struct ShallowWaterModelBase {
  MPASMesh *m_mesh;
  Real m_grav;
  bool m_disable_h_tendency;
  bool m_disable_vn_tendency;
  Int m_ntracers;
  Real1d m_f_vertex;
  Real1d m_f_edge;

  virtual void compute_auxiliary_variables(RealConst2d h_cell,
                                           RealConst2d vn_edge,
                                           RealConst3d tr_cell) const;

  virtual void compute_h_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                                  RealConst2d vn_edge,
                                  AddMode add_mode) const = 0;
  virtual void compute_vn_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                                   RealConst2d vn_edge,
                                   AddMode add_mode) const = 0;
  virtual void compute_tr_tendency(Real3d tr_tend_cell, RealConst3d tr_cell,
                                   RealConst2d vn_edge,
                                   AddMode add_mode) const = 0;
  virtual void additional_tendency(Real2d h_tend_cell, Real2d vn_tend_edge,
                                   RealConst2d h_cell, RealConst2d vn_edge,
                                   Real t) const {}
  virtual Real mass_integral(RealConst2d h_cell) const;
  virtual Real circulation_integral(RealConst2d vn_edge) const;
  virtual Real energy_integral(RealConst2d h_cell,
                               RealConst2d vn_edge) const = 0;

  void compute_tendency(const ShallowWaterState &tend,
                        const ShallowWaterState &state, Real t,
                        AddMode add_mode = AddMode::replace) const;

  ShallowWaterModelBase(MPASMesh *mesh, const ShallowWaterParams &params);
};

struct ShallowWaterModel : ShallowWaterModelBase {
  Real m_drag_coeff;
  Real m_visc_del2;
  Real m_visc_del4;
  Real m_eddy_diff2;
  Real m_eddy_diff4;

  Real2d m_ke_cell;
  Real2d m_div_cell;
  Real3d m_norm_tr_cell;

  Real2d m_h_flux_edge;
  Real2d m_h_mean_edge;
  Real2d m_h_drag_edge;
  Real2d m_norm_rvort_edge;
  Real2d m_norm_f_edge;

  Real2d m_rvort_vertex;
  Real2d m_norm_rvort_vertex;
  Real2d m_norm_f_vertex;

  void compute_auxiliary_variables(RealConst2d h_cell, RealConst2d vn_edge,
                                   RealConst3d tr_cell) const override;
  void compute_cell_auxiliary_variables(RealConst2d h_cell, RealConst2d vn_edge,
                                        RealConst3d tr_cell) const;
  void compute_edge_auxiliary_variables(RealConst2d h_cell, RealConst2d vn_edge,
                                        RealConst3d tr_cell) const;
  void compute_vertex_auxiliary_variables(RealConst2d h_cell,
                                          RealConst2d vn_edge,
                                          RealConst3d tr_cell) const;

  void compute_h_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                          RealConst2d vn_edge, AddMode add_mode) const override;
  void compute_vn_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override;
  void compute_tr_tendency(Real3d tr_tend_cell, RealConst3d tr_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override;
  Real energy_integral(RealConst2d h, RealConst2d v) const override;

  ShallowWaterModel(MPASMesh *mesh, const ShallowWaterParams &params);
};

struct LinearShallowWaterModel : ShallowWaterModelBase {
  Real m_h0;
  void compute_h_tendency(Real2d h_tend_cell, RealConst2d h_cell,
                          RealConst2d vn_edge, AddMode add_mode) const override;
  void compute_vn_tendency(Real2d vn_tend_edge, RealConst2d h_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override;
  void compute_tr_tendency(Real3d tr_tend_cell, RealConst3d tr_cell,
                           RealConst2d vn_edge,
                           AddMode add_mode) const override {}
  Real energy_integral(RealConst2d h_cell, RealConst2d vn_edge) const override;

  LinearShallowWaterModel(MPASMesh *mesh,
                          const LinearShallowWaterParams &params);
};

class DivergenceCell {
  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Real2d m_edge_sign_on_cell;
  Real1d m_dv_edge;
  Real1d m_area_cell;

  public:

  YAKL_INLINE Real operator()(Int icell, Int k,
                              const RealConst2d &v_edge) const {
    Real accum = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      Int jedge = m_edges_on_cell(icell, j);
      accum += m_dv_edge(jedge) * m_edge_sign_on_cell(icell, j) * v_edge(jedge, k);
    }
    Real inv_area_cell = 1._fp / m_area_cell(icell);
    return accum * inv_area_cell;
  }
  
  YAKL_INLINE Real operator()(Int icell, Int k,
                              const RealConst2d &v_edge,
                              const RealConst2d &h_edge) const {
    Real accum = 0;
    for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
      Int jedge = m_edges_on_cell(icell, j);
      accum += m_dv_edge(jedge) * m_edge_sign_on_cell(icell, j) * h_edge(jedge, k) * v_edge(jedge, k);
    }
    Real inv_area_cell = 1._fp / m_area_cell(icell);
    return accum * inv_area_cell;
  }

  DivergenceCell(const MPASMesh *mesh) :
    m_nedges_on_cell(mesh->m_nedges_on_cell),
    m_edges_on_cell(mesh->m_edges_on_cell),
    m_edge_sign_on_cell(mesh->m_edge_sign_on_cell),
    m_dv_edge(mesh->m_dv_edge),
    m_area_cell(mesh->m_area_cell) {
  }
};

struct Del2Std {};
struct Del2Mod {};

class Del2UEdge {
  Int2d m_cells_on_edge;
  Int2d m_vertices_on_edge;
  Real1d m_dc_edge;
  Real1d m_dv_edge;

  public:
  
  YAKL_INLINE Real operator()(Int iedge, Int k,
                              const RealConst2d &div_cell,
                              const RealConst2d &rvort_vertex, Del2Std) const {
      Int icell0 = m_cells_on_edge(iedge, 0);
      Int icell1 = m_cells_on_edge(iedge, 1);

      Int ivertex0 = m_vertices_on_edge(iedge, 0);
      Int ivertex1 = m_vertices_on_edge(iedge, 1);

      Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
      Real dv_edge_inv =
          1._fp / m_dv_edge(iedge);

      Real del2u =
          ((div_cell(icell1, k) - div_cell(icell0, k)) * dc_edge_inv -
           (rvort_vertex(ivertex1, k) - rvort_vertex(ivertex0, k)) *
               dv_edge_inv);

      return del2u;
  }

  YAKL_INLINE Real operator()(Int iedge, Int k,
                              const RealConst2d &div_cell,
                              const RealConst2d &rvort_vertex, Del2Mod) const {
      Int icell0 = m_cells_on_edge(iedge, 0);
      Int icell1 = m_cells_on_edge(iedge, 1);

      Int ivertex0 = m_vertices_on_edge(iedge, 0);
      Int ivertex1 = m_vertices_on_edge(iedge, 1);

      Real dc_edge_inv = 1._fp / m_dc_edge(iedge);
      Real dv_edge_inv =
          1._fp / std::max(m_dv_edge(iedge), 0.25_fp * m_dc_edge(iedge)); // huh

      Real del2u =
          ((div_cell(icell1, k) - div_cell(icell0, k)) * dc_edge_inv -
           (rvort_vertex(ivertex1, k) - rvort_vertex(ivertex0, k)) *
               dv_edge_inv);

      return del2u;
  }

  Del2UEdge(const MPASMesh *mesh) :
    m_cells_on_edge(mesh->m_cells_on_edge),
    m_vertices_on_edge(mesh->m_vertices_on_edge),
    m_dc_edge(mesh->m_dc_edge),
    m_dv_edge(mesh->m_dv_edge) {
  }
};

class VorticityVertex {
  Int2d m_edges_on_vertex;
  Real2d m_edge_sign_on_vertex;
  Real1d m_dc_edge;
  Real1d m_area_triangle;

  public:
  
  YAKL_INLINE Real operator()(Int ivertex, Int k,
                              const RealConst2d &vn_edge) const {
     Real del2rvort = -0;
     for (Int j = 0; j < 3; ++j) {
       Int jedge = m_edges_on_vertex(ivertex, j);
       del2rvort += m_dc_edge(jedge) * m_edge_sign_on_vertex(ivertex, j) *
                    vn_edge(jedge, k);
     }
     Real inv_area_triangle = 1._fp / m_area_triangle(ivertex);
     del2rvort *= inv_area_triangle;

     return del2rvort;
  }

  VorticityVertex(const MPASMesh *mesh) :
    m_edges_on_vertex(mesh->m_edges_on_vertex),
    m_edge_sign_on_vertex(mesh->m_edge_sign_on_vertex),
    m_dc_edge(mesh->m_dc_edge),
    m_area_triangle(mesh->m_area_triangle) {
  }
};

class ThicknessVertex {
  Int2d m_cells_on_vertex;
  Real2d m_kiteareas_on_vertex;
  Real1d m_area_triangle;


  public:
  
  YAKL_INLINE Real operator()(Int ivertex, Int k,
                              const RealConst2d &h_cell) const {
    Real h = -0;
    Real inv_area_triangle = 1._fp / m_area_triangle(ivertex);
    for (Int j = 0; j < 3; ++j) {
      Int jcell = m_cells_on_vertex(ivertex, j);
      h += m_kiteareas_on_vertex(ivertex, j) * h_cell(jcell, k);
    }
    h *= inv_area_triangle;
    return h;
  }

  ThicknessVertex(const MPASMesh *mesh) :
    m_cells_on_vertex(mesh->m_cells_on_vertex),
    m_kiteareas_on_vertex(mesh->m_kiteareas_on_vertex),
    m_area_triangle(mesh->m_area_triangle) {
  }
};

class QTermEdge {
  Int1d m_nedges_on_edge;
  Int2d m_edges_on_edge;
  Real2d m_weights_on_edge;

  public:
  
  YAKL_INLINE Real operator()(Int iedge, Int k,
                              const RealConst2d &norm_rvort_edge,
                              const RealConst2d &norm_f_edge,
                              const RealConst2d &h_flux_edge,
                              const RealConst2d &vn_edge) const {
     Real qt = -0;
     for (Int j = 0; j < m_nedges_on_edge(iedge); ++j) {
       Int jedge = m_edges_on_edge(iedge, j);

       Real norm_vort = (norm_rvort_edge(iedge, k) + norm_f_edge(iedge, k) +
                         norm_rvort_edge(jedge, k) + norm_f_edge(jedge, k)) *
                        0.5_fp;

       qt += m_weights_on_edge(iedge, j) * h_flux_edge(jedge, k) *
             vn_edge(jedge, k) * norm_vort;
     }

     return qt;
  }

  QTermEdge(const MPASMesh *mesh) :
    m_nedges_on_edge(mesh->m_nedges_on_edge),
    m_edges_on_edge(mesh->m_edges_on_edge),
    m_weights_on_edge(mesh->m_weights_on_edge) {
  }
};

class GradEdge {
  Int2d m_cells_on_edge;
  Real1d m_dc_edge;

  public:
  
  YAKL_INLINE Real operator()(Int iedge, Int k,
                              const RealConst2d &ke_cell) const {
     Int icell0 = m_cells_on_edge(iedge, 0);
     Int icell1 = m_cells_on_edge(iedge, 1);
     Real inv_dc_edge = 1._fp / m_dc_edge(iedge);
     return (ke_cell(icell1, k) - ke_cell(icell0, k)) * inv_dc_edge;
  }

  GradEdge(const MPASMesh *mesh) :
    m_cells_on_edge(mesh->m_cells_on_edge),
    m_dc_edge(mesh->m_dc_edge) {
  }
};

class TracerDel2Cell {
  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Int2d m_cells_on_edge;
  Real2d m_edge_sign_on_cell;
  Real1d m_mesh_scaling_del2;
  Real1d m_dc_edge;
  Real1d m_dv_edge;
  Real1d m_area_cell;

  public:
  
  YAKL_INLINE Real operator()(Int l, Int icell, Int k,
                              const RealConst3d &norm_tr_cell,
                              const RealConst2d &h_mean_edge
                              ) const {
     Real tr_del2 = -0;
     for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
       Int jedge = m_edges_on_cell(icell, j);
       Int jcell0 = m_cells_on_edge(jedge, 0);
       Int jcell1 = m_cells_on_edge(jedge, 1);

       Real inv_dc_edge = 1._fp / m_dc_edge(jedge);
       Real grad_tr_edge =
           (norm_tr_cell(l, jcell1, k) - norm_tr_cell(l, jcell0, k)) *
           inv_dc_edge;

       tr_del2 += m_dv_edge(jedge) * m_edge_sign_on_cell(icell, j) *
                  h_mean_edge(jedge, k) * m_mesh_scaling_del2(jedge) * grad_tr_edge;
     }
     Real inv_area_cell = 1._fp / m_area_cell(icell);
     return tr_del2 * inv_area_cell;
  }

  TracerDel2Cell(const MPASMesh *mesh) :
    m_nedges_on_cell(mesh->m_nedges_on_cell),
    m_edges_on_cell(mesh->m_edges_on_cell),
    m_cells_on_edge(mesh->m_cells_on_edge),
    m_edge_sign_on_cell(mesh->m_edge_sign_on_cell),
    m_mesh_scaling_del2(mesh->m_mesh_scaling_del2),
    m_dc_edge(mesh->m_dc_edge),
    m_dv_edge(mesh->m_dv_edge), 
    m_area_cell(mesh->m_area_cell) {
  }
};

class TracerAdvFluxEdge {
  Int2d m_cells_on_edge;
  public:
  
  YAKL_INLINE Real operator()(Int l, Int jedge, Int k,
                              const RealConst3d &norm_tr_cell,
                              const RealConst2d &h_flux_edge,
                              const RealConst2d &vn_edge
                              ) const {
     Int jcell0 = m_cells_on_edge(jedge, 0);
     Int jcell1 = m_cells_on_edge(jedge, 1);

     Real norm_tr_edge =
         (norm_tr_cell(l, jcell0, k) + norm_tr_cell(l, jcell1, k)) *
         0.5_fp;

     // advection
     Real tr_flux =
         -h_flux_edge(jedge, k) * norm_tr_edge * vn_edge(jedge, k);
     return tr_flux;
  }

  TracerAdvFluxEdge(const MPASMesh *mesh) :
    m_cells_on_edge(mesh->m_cells_on_edge) {
  }
};

class TracerDel2FluxEdge {
  Int2d m_cells_on_edge;
  Real1d m_dc_edge;
  public:
  
  YAKL_INLINE Real operator()(Int l, Int jedge, Int k,
                              const RealConst3d &norm_tr_cell,
                              const RealConst2d &h_mean_edge,
                              Real eddy_diff2
                              ) const {
    
    Int jcell0 = m_cells_on_edge(jedge, 0);
    Int jcell1 = m_cells_on_edge(jedge, 1);

    Real inv_dc_edge = 1._fp / m_dc_edge(jedge);
    Real grad_tr_edge =
        (norm_tr_cell(l, jcell1, k) - norm_tr_cell(l, jcell0, k)) *
        inv_dc_edge;
    return eddy_diff2 * h_mean_edge(jedge, k) * grad_tr_edge;
  }

  TracerDel2FluxEdge(const MPASMesh *mesh) :
    m_cells_on_edge(mesh->m_cells_on_edge),
    m_dc_edge(mesh->m_dc_edge) {
  }
};

class TracerDel4FluxEdge {
  Int2d m_cells_on_edge;
  Real1d m_dc_edge;
  Real1d m_mesh_scaling_del4;
  public:
  
  YAKL_INLINE Real operator()(Int l, Int jedge, Int k,
                              const RealConst3d &tmp_tr_del2_cell,
                              Real eddy_diff4
                              ) const {
    
    Int jcell0 = m_cells_on_edge(jedge, 0);
    Int jcell1 = m_cells_on_edge(jedge, 1);

    Real inv_dc_edge = 1._fp / m_dc_edge(jedge);
    Real grad_tr_del2_edge = (tmp_tr_del2_cell(l, jcell1, k) -
                              tmp_tr_del2_cell(l, jcell0, k)) *
                             inv_dc_edge;
    return -eddy_diff4 * grad_tr_del2_edge * m_mesh_scaling_del4(jedge);
  }

  TracerDel4FluxEdge(const MPASMesh *mesh) :
    m_cells_on_edge(mesh->m_cells_on_edge),
    m_dc_edge(mesh->m_dc_edge),
    m_mesh_scaling_del4(mesh->m_mesh_scaling_del4) {
  }
};

class KineticEnergyCell {
  Int1d m_nedges_on_cell;
  Int2d m_edges_on_cell;
  Real1d m_dc_edge;
  Real1d m_dv_edge;
  Real1d m_area_cell;

  public:
  
  YAKL_INLINE Real operator()(Int icell, Int k,
                              const RealConst2d &vn_edge
                              ) const {
     Real ke = -0;
     for (Int j = 0; j < m_nedges_on_cell(icell); ++j) {
       Int jedge = m_edges_on_cell(icell, j);
       Real area_edge = m_dv_edge(jedge) * m_dc_edge(jedge);
       ke += area_edge * vn_edge(jedge, k) * vn_edge(jedge, k) * 0.25_fp;
     }
     Real inv_area_cell = 1._fp / m_area_cell(icell);
     ke *= inv_area_cell;
     return ke;
  }

  KineticEnergyCell(const MPASMesh *mesh) :
    m_nedges_on_cell(mesh->m_nedges_on_cell),
    m_edges_on_cell(mesh->m_edges_on_cell),
    m_dc_edge(mesh->m_dc_edge),
    m_dv_edge(mesh->m_dv_edge), 
    m_area_cell(mesh->m_area_cell) {
  }
};

class CellAverageEdge {
  Int2d m_cells_on_edge;
  public:
  
  YAKL_INLINE Real operator()(Int iedge, Int k,
                              const RealConst2d &h_cell) const {
    Int jcell0 = m_cells_on_edge(iedge, 0);
    Int jcell1 = m_cells_on_edge(iedge, 1);
    return 0.5_fp * (h_cell(jcell0, k) + h_cell(jcell1, k));
  }

  CellAverageEdge(const MPASMesh *mesh) :
    m_cells_on_edge(mesh->m_cells_on_edge) {
  }
};

class VertexAverageEdge {
  Int2d m_vertices_on_edge;

  public:
  
  YAKL_INLINE Real operator()(Int iedge, Int k,
                              const RealConst2d &h_vertex) const {
    Int jvertex0 = m_vertices_on_edge(iedge, 0);
    Int jvertex1 = m_vertices_on_edge(iedge, 1);
    return 0.5_fp * (h_vertex(jvertex0, k) + h_vertex(jvertex1, k));
  }

  VertexAverageEdge(const MPASMesh *mesh) :
    m_vertices_on_edge(mesh->m_vertices_on_edge) {
  }
};

} // namespace omega
