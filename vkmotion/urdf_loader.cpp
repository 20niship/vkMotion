#include "urdf_loader.hpp"

#include <tinyxml2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cutil/mat.hpp>
#include <cutil/quaternion.hpp>
#include <stdexcept>

namespace vkmotion {

namespace {

cutil::Vec3f parse_vec3(const char* s, cutil::Vec3f fallback = cutil::Vec3f(0.0f, 0.0f, 0.0f)) {
  if(!s) return fallback;
  float v[3] = {0.0f, 0.0f, 0.0f};
  std::sscanf(s, "%f %f %f", &v[0], &v[1], &v[2]);
  return cutil::Vec3f(v[0], v[1], v[2]);
}

UrdfJointType parse_joint_type(const char* s) {
  std::string t = s ? s : "fixed";
  if(t == "revolute") return UrdfJointType::kRevolute;
  if(t == "continuous") return UrdfJointType::kContinuous;
  if(t == "prismatic") return UrdfJointType::kPrismatic;
  return UrdfJointType::kFixed;
}

void parse_origin(const tinyxml2::XMLElement* parent, cutil::Vec3f& xyz, cutil::Vec3f& rpy) {
  xyz = cutil::Vec3f(0.0f, 0.0f, 0.0f);
  rpy = cutil::Vec3f(0.0f, 0.0f, 0.0f);
  const tinyxml2::XMLElement* origin = parent->FirstChildElement("origin");
  if(!origin) return;
  xyz = parse_vec3(origin->Attribute("xyz"));
  rpy = parse_vec3(origin->Attribute("rpy"));
}

bool parse_collision(const tinyxml2::XMLElement* collision_el, UrdfCollisionGeometry& out) {
  parse_origin(collision_el, out.origin_xyz, out.origin_rpy);
  const tinyxml2::XMLElement* geom = collision_el->FirstChildElement("geometry");
  if(!geom) return false;
  if(const tinyxml2::XMLElement* box = geom->FirstChildElement("box")) {
    out.type     = UrdfGeometryType::kBox;
    out.box_size = parse_vec3(box->Attribute("size"));
    return true;
  }
  if(const tinyxml2::XMLElement* cyl = geom->FirstChildElement("cylinder")) {
    out.type = UrdfGeometryType::kCylinder;
    cyl->QueryFloatAttribute("radius", &out.radius);
    cyl->QueryFloatAttribute("length", &out.length);
    return true;
  }
  if(const tinyxml2::XMLElement* sph = geom->FirstChildElement("sphere")) {
    out.type = UrdfGeometryType::kSphere;
    sph->QueryFloatAttribute("radius", &out.radius);
    return true;
  }
  if(const tinyxml2::XMLElement* cap = geom->FirstChildElement("capsule")) {
    out.type = UrdfGeometryType::kCapsule;
    cap->QueryFloatAttribute("radius", &out.radius);
    cap->QueryFloatAttribute("length", &out.length);
    return true;
  }
  // <mesh> unsupported: REQUIREMENTS.md keeps this repo mesh-free (primitive-only collision geometry).
  return false;
}

}  // namespace

int UrdfModel::find_link(const std::string& link_name) const {
  for(size_t i = 0; i < links.size(); i++) {
    if(links[i].name == link_name) return static_cast<int>(i);
  }
  return -1;
}

UrdfModel load_urdf(const std::string& path) {
  tinyxml2::XMLDocument doc;
  if(doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("vkmotion: failed to load URDF file: " + path);
  }
  const tinyxml2::XMLElement* robot = doc.RootElement();
  if(!robot || std::string(robot->Name()) != "robot") {
    throw std::runtime_error("vkmotion: not a <robot> URDF document: " + path);
  }

  UrdfModel model;
  model.name = robot->Attribute("name") ? robot->Attribute("name") : "";

  for(const tinyxml2::XMLElement* link_el = robot->FirstChildElement("link"); link_el;
      link_el                             = link_el->NextSiblingElement("link")) {
    UrdfLink link;
    link.name = link_el->Attribute("name") ? link_el->Attribute("name") : "";
    for(const tinyxml2::XMLElement* collision_el = link_el->FirstChildElement("collision"); collision_el;
        collision_el                             = collision_el->NextSiblingElement("collision")) {
      UrdfCollisionGeometry geom;
      if(parse_collision(collision_el, geom)) link.collisions.push_back(geom);
    }
    model.links.push_back(std::move(link));
  }

  for(const tinyxml2::XMLElement* joint_el = robot->FirstChildElement("joint"); joint_el;
      joint_el                             = joint_el->NextSiblingElement("joint")) {
    UrdfJoint joint;
    joint.name = joint_el->Attribute("name") ? joint_el->Attribute("name") : "";
    joint.type = parse_joint_type(joint_el->Attribute("type"));
    parse_origin(joint_el, joint.origin_xyz, joint.origin_rpy);

    if(const tinyxml2::XMLElement* parent_el = joint_el->FirstChildElement("parent")) {
      joint.parent_link = parent_el->Attribute("link") ? parent_el->Attribute("link") : "";
    }
    if(const tinyxml2::XMLElement* child_el = joint_el->FirstChildElement("child")) {
      joint.child_link = child_el->Attribute("link") ? child_el->Attribute("link") : "";
    }
    if(const tinyxml2::XMLElement* axis_el = joint_el->FirstChildElement("axis")) {
      joint.axis = parse_vec3(axis_el->Attribute("xyz"), cutil::Vec3f(1.0f, 0.0f, 0.0f));
    }
    if(const tinyxml2::XMLElement* limit_el = joint_el->FirstChildElement("limit")) {
      limit_el->QueryFloatAttribute("lower", &joint.limit_lower);
      limit_el->QueryFloatAttribute("upper", &joint.limit_upper);
      limit_el->QueryFloatAttribute("velocity", &joint.limit_velocity);
    }
    model.joints.push_back(std::move(joint));
  }

  return model;
}

std::vector<std::string> active_joint_names(const UrdfModel& model) {
  std::vector<std::string> names;
  for(const auto& j : model.joints) {
    if(j.type != UrdfJointType::kFixed) names.push_back(j.name);
  }
  return names;
}

namespace {

// Smallest number of same-size spheres that stack end-to-end to cover `length` along one axis.
int sphere_count_for_length(float length, float sphere_diameter) {
  if(sphere_diameter <= 0.0f) return 1;
  return std::max(1, static_cast<int>(std::ceil(length / sphere_diameter)));
}

std::vector<CollisionSphere> spheres_for_geometry(const std::string& link_name, const UrdfCollisionGeometry& geom) {
  std::vector<CollisionSphere> out;
  // geometry-local origin offset (<collision><origin>), spheres are expressed in the link frame.
  const cutil::Mat4f origin = cutil::mat4_translation(geom.origin_xyz) * cutil::mat4_rotation(cutil::Quatf::from_euler(geom.origin_rpy));

  switch(geom.type) {
    case UrdfGeometryType::kSphere: {
      CollisionSphere s;
      s.link_name = link_name;
      s.center    = geom.origin_xyz;
      s.radius    = geom.radius;
      out.push_back(s);
      break;
    }
    case UrdfGeometryType::kCylinder:
    case UrdfGeometryType::kCapsule: {
      int n = sphere_count_for_length(geom.length, 2.0f * geom.radius);
      for(int i = 0; i < n; i++) {
        float t  = (n == 1) ? 0.0f : (static_cast<float>(i) / static_cast<float>(n - 1) - 0.5f) * geom.length;
        cutil::Vec3f local(0.0f, 0.0f, t);
        CollisionSphere s;
        s.link_name = link_name;
        s.center    = (origin * cutil::NVec<4, float>(local, 1.0f)).xyz();
        s.radius    = geom.radius;
        out.push_back(s);
      }
      break;
    }
    case UrdfGeometryType::kBox: {
      float r = std::min({geom.box_size[0], geom.box_size[1], geom.box_size[2]}) * 0.5f;
      if(r <= 1e-6f) r = 1e-3f;
      int nx = sphere_count_for_length(geom.box_size[0], 2.0f * r);
      int ny = sphere_count_for_length(geom.box_size[1], 2.0f * r);
      int nz = sphere_count_for_length(geom.box_size[2], 2.0f * r);
      for(int ix = 0; ix < nx; ix++) {
        for(int iy = 0; iy < ny; iy++) {
          for(int iz = 0; iz < nz; iz++) {
            auto axis_offset = [](int i, int n, float extent) {
              return (n == 1) ? 0.0f : (static_cast<float>(i) / static_cast<float>(n - 1) - 0.5f) * extent;
            };
            cutil::Vec3f local(axis_offset(ix, nx, geom.box_size[0]), axis_offset(iy, ny, geom.box_size[1]), axis_offset(iz, nz, geom.box_size[2]));
            CollisionSphere s;
            s.link_name = link_name;
            s.center    = (origin * cutil::NVec<4, float>(local, 1.0f)).xyz();
            s.radius    = r;
            out.push_back(s);
          }
        }
      }
      break;
    }
    case UrdfGeometryType::kMesh:
      break;
  }
  return out;
}

}  // namespace

std::vector<CollisionSphere> generate_collision_spheres_from_urdf(const UrdfModel& model) {
  std::vector<CollisionSphere> spheres;
  for(const auto& link : model.links) {
    for(const auto& geom : link.collisions) {
      auto link_spheres = spheres_for_geometry(link.name, geom);
      spheres.insert(spheres.end(), link_spheres.begin(), link_spheres.end());
    }
  }
  return spheres;
}

}  // namespace vkmotion
