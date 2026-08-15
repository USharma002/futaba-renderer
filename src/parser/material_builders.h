#pragma once

#include <vector>
#include <string>
#include "proplist.h"
#include "material.cuh"

FUTABA_NAMESPACE_BEGIN

Material make_diffuse_material(const PropertyList& bsdfProps,
                              const PropertyList& emitterProps = PropertyList());

Material make_dielectric_material(const PropertyList& bsdfProps,
                                  const PropertyList& emitterProps = PropertyList());

Material make_mirror_material(const PropertyList& bsdfProps,
                              const PropertyList& emitterProps = PropertyList());

Material make_microfacet_material(const PropertyList& bsdfProps,
                                  const PropertyList& emitterProps = PropertyList());

Material make_roughplastic_material(const PropertyList& bsdfProps,
                                    const PropertyList& emitterProps = PropertyList());

Material make_roughdielectric_material(const PropertyList& bsdfProps,
                                       const PropertyList& emitterProps = PropertyList());

Material make_roughconductor_material(const PropertyList& bsdfProps,
                                      const PropertyList& emitterProps = PropertyList());

Material make_thindielectric_material(const PropertyList& bsdfProps,
                                      const PropertyList& emitterProps = PropertyList());

Material makeMaterialFromPropertyLists(
        const PropertyList&       bsdfProps,
        const PropertyList&       emitterProps,
        std::vector<std::string>& warnings);

Material makeMaterialFromPropertyLists(const PropertyList& bsdfProps,
                                      const PropertyList& emitterProps);

FUTABA_NAMESPACE_END