"""Numerical regression tests and malformed-input tests; no network needed."""
from pathlib import Path
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
import warnings
import numpy as np
import yaml

# Run directly from the source checkout as well as an installed package.
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from optical_materials import (load_material, available_materials, save_material, register_loader,
    to_omega, from_omega, nk_to_epsilon, epsilon_to_nk, sigma_to_epsilon, epsilon_to_sigma,
    MaterialError, MaterialWarning, RangeError, ResponseUnavailableError)
from optical_materials.units import C, H, HBAR, E_CHARGE, EPSILON_0
from optical_materials.responses import Oscillator, DrudeLorentzParameters, evaluate_drude_lorentz
from optical_materials.io import BUILTIN_DATABASE


class UnitsTests(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(float(to_omega(1, "Hz")), 2*np.pi)
        self.assertAlmostEqual(float(to_omega(1, "THz")), 2*np.pi*1e12)
        self.assertAlmostEqual(float(to_omega(1, "eV"))/1e15, 1.519267447878626)
        self.assertAlmostEqual(float(to_omega(1, "cm-1")), 2*np.pi*C*100)

    def test_all_unit_round_trips(self):
        for unit in ["eV","Hz","THz","cm-1","rad/s"]:
            with self.subTest(unit=unit):
                x=np.array([[1.,2.],[3.,4.]])
                np.testing.assert_allclose(from_omega(to_omega(x, unit), unit),x,rtol=2e-15)

    def test_hz_not_angular(self):
        self.assertAlmostEqual(float(to_omega(2, "Hz") / to_omega(2, "rad/s")), 2*np.pi)

    def test_bad_unit(self):
        for unit in ["GHz","hz",None, ["Hz"]]:
            with self.subTest(unit=unit), self.assertRaises(MaterialError):
                to_omega(1,unit)

    def test_bad_coordinates(self):
        for value in [0,-1,np.nan,np.inf,[],1+0j,"bad"]:
            with self.subTest(value=value), self.assertRaises(MaterialError):
                to_omega(value,"Hz")


class ConversionTests(unittest.TestCase):
    def test_nk_known_value(self):
        self.assertEqual(nk_to_epsilon(2, .5),3.75+2j)

    def test_nk_inverse(self):
        n=np.array([1.5,2.,.1]); k=np.array([0,.3,4.])
        actual=epsilon_to_nk(nk_to_epsilon(n,k))
        np.testing.assert_allclose(actual[0],n)
        np.testing.assert_allclose(actual[1],k)

    def test_bulk_sigma_sign_and_inverse(self):
        omega=np.array([1e13,2e13,4e13]); sigma=np.array([100+50j,50+10j,20-5j])
        eps=sigma_to_epsilon(sigma,omega,epsilon_bg=2)
        np.testing.assert_allclose(eps.real,2-sigma.imag/(EPSILON_0*omega))
        np.testing.assert_allclose(eps.imag,sigma.real/(EPSILON_0*omega))
        np.testing.assert_allclose(epsilon_to_sigma(eps,omega,2),sigma,rtol=1e-13)

    def test_sheet_thickness_conversion(self):
        actual=sigma_to_epsilon(1e-3+2e-4j,1e14,thickness_m=2e-9)
        expected=sigma_to_epsilon((1e-3+2e-4j)/2e-9,1e14)
        np.testing.assert_allclose(actual,expected)

    def test_invalid_thickness(self):
        for t in [0,-1,np.inf,[1e-9,2e-9]]:
            with self.subTest(t=t), self.assertRaises(MaterialError):
                sigma_to_epsilon(.1,1e14,thickness_m=t)

    def test_nonfinite_response(self):
        for function,args in [(nk_to_epsilon,(np.nan,1)),(epsilon_to_nk,(np.inf,)),
                              (sigma_to_epsilon,(np.nan,1e14)),(epsilon_to_sigma,(np.inf,1e14))]:
            with self.subTest(function=function.__name__), self.assertRaises(MaterialError):
                function(*args)

    def test_gain_square_root_branch(self):
        n,k=epsilon_to_nk(3-4j)
        self.assertEqual(n,2)
        self.assertEqual(k,-1)


class PublicApiTests(unittest.TestCase):
    def test_all_examples_load_and_evaluate(self):
        self.assertTrue({'Au', 'Ag', 'SiO2', 'WTe2', 'DemoNK', 'DemoEpsilon',
                         'DemoBulkSigma', 'DemoBiaxial'}.issubset(available_materials()))
        for name in available_materials():
            with self.subTest(name=name):
                mat=load_material(name)
                w=np.geomspace(*mat.valid_frequency_range,21)
                fn=mat.sigma if mat.dimensionality == 2 else mat.epsilon
                value=fn(w)
                self.assertTrue(np.isfinite(value).all())
                self.assertEqual(value.shape,(21,mat.dimensionality,mat.dimensionality))

    def test_isotropic_tensor_shapes(self):
        mat=load_material("Au")
        w=to_omega([[1.,2.],[3.,4.]],"eV")
        tensor=mat.epsilon(w)
        self.assertEqual(tensor.shape,(2,2,3,3))
        self.assertEqual(mat.epsilon(to_omega(1.,"eV")).shape,(3,3))
        for c,i in [("x",0),("y",1),("z",2)]:
            np.testing.assert_allclose(tensor[...,i,i],mat.epsilon(w,component=c))
        self.assertTrue(np.all(tensor[...,0,1] == 0))

    def test_sheet_tensor_and_film(self):
        mat=load_material("WTe2"); w=to_omega([.1,.2],"eV")
        self.assertEqual(mat.sigma(w).shape,(2,2,2))
        self.assertFalse(np.allclose(mat.sigma(w,"x"),mat.sigma(w,"y")))
        with self.assertRaises(ResponseUnavailableError): mat.epsilon(w)
        film=mat.epsilon(w,thickness_m=1e-9,epsilon_bg=2.)
        self.assertEqual(film.shape,(2,3,3))
        np.testing.assert_allclose(film[:,2,2],2.)
        np.testing.assert_allclose(film[:,0,0],sigma_to_epsilon(mat.sigma(w,"x"),w,2.,thickness_m=1e-9))

    def test_sheet_z_rejects_bad_thickness(self):
        with self.assertRaises(MaterialError): load_material("WTe2").epsilon(to_omega(.1,"eV"),"z",thickness_m=-1)

    def test_bulk_rejects_sheet_thickness(self):
        with self.assertRaises(MaterialError): load_material("Au").epsilon(to_omega(1,"eV"),thickness_m=1e-9)

    def test_biaxial_component_layout(self):
        mat=load_material("DemoBiaxial"); w=to_omega([2.,5.],"THz")
        tensor=mat.epsilon(w)
        for index,c in enumerate("xyz"):
            np.testing.assert_allclose(tensor[:,index,index],mat.epsilon(w,c))
        self.assertTrue(np.all(tensor[:,0,0] != tensor[:,1,1]))

    def test_equivalent_formats_between_samples(self):
        nk=load_material("DemoNK"); eps=load_material("DemoEpsilon")
        w=to_omega(np.linspace(.501,2.999,137),"eV")
        np.testing.assert_allclose(nk.epsilon(w),eps.epsilon(w),rtol=5e-14,atol=5e-14)

    def test_interpolation_independent_parts(self):
        mat=load_material("DemoNK"); grid=mat.omega
        w=(grid[3]+grid[4])/2
        expected=(mat.epsilon(grid[3],"x")+mat.epsilon(grid[4],"x"))/2
        np.testing.assert_allclose(mat.epsilon(w,"x"),expected,rtol=2e-15)

    def test_invalid_components(self):
        for name,c in [("Au","q"),("WTe2","z"),("DemoBiaxial","iso")]:
            with self.subTest(name=name),self.assertRaises(MaterialError):
                load_material(name).sigma(to_omega(1,"eV") if name == "Au" else np.mean(load_material(name).valid_frequency_range),c)

    def test_bounds_for_models_and_tables(self):
        for name in ["Au","DemoNK","WTe2"]:
            mat=load_material(name); lo,hi=mat.valid_frequency_range
            for w in [lo*.999,hi*1.001]:
                with self.subTest(name=name,w=w),self.assertRaises(RangeError): mat.sigma(w)

    def test_query_validation(self):
        for w in [0,-1,np.nan,np.inf,[],1+0j]:
            with self.subTest(w=w),self.assertRaises(MaterialError): load_material("Au").epsilon(w)

    def test_metadata_and_parameters_are_isolated(self):
        mat=load_material("Au"); meta=mat.metadata; meta["name"]="other";meta["components"].append("q")
        self.assertEqual(mat.name,"Au")
        self.assertEqual(mat.metadata["components"],["iso"])
        with self.assertRaises(TypeError): mat.parameters["iso"] = None
        self.assertEqual(mat.metadata["internal_time_convention"],"exp(-i*omega*t)")
        self.assertEqual(mat.metadata["data_type"],"model")

    def test_bulk_native_and_effective_sigma(self):
        mat=load_material("DemoBulkSigma"); w=to_omega(3e12,"Hz")
        np.testing.assert_allclose(mat.epsilon(w,"x"),sigma_to_epsilon(mat.sigma(w,"x"),w))
        with self.assertRaises(MaterialError):mat.sigma(w,epsilon_bg=2)
        au=load_material("Au");w=to_omega(2,"eV")
        np.testing.assert_allclose(au.sigma(w,"x",epsilon_bg=2),epsilon_to_sigma(au.epsilon(w,"x"),w,2))

    def test_native_epsilon_rejects_background_override(self):
        with self.assertRaisesRegex(MaterialError,"already provides permittivity"):
            load_material("Au").epsilon(to_omega(1,"eV"),epsilon_bg=2)

    def test_reference_au_ag_equation(self):
        # Independent evaluation in eV, from the cited published LD parameterization.
        e=np.array([.5,1.,2.,4.])
        specs=[("Au",9.03,.760,.053,[(.024,.415,.241),(.010,.830,.345),(.071,2.969,.870),(.601,4.304,2.494),(4.384,13.32,2.214)]),
               ("Ag",9.01,.845,.048,[(.065,.816,3.886),(.124,4.481,.452),(.011,8.185,.065),(.840,9.083,.916),(5.646,20.29,2.419)])]
        for name,wp,f0,g0,terms in specs:
            expected=1-f0*wp**2/(e*(e+1j*g0))
            for f,w,g in terms:expected+=f*wp**2/(w*w-e*e-1j*g*e)
            np.testing.assert_allclose(load_material(name).epsilon(to_omega(e,"eV"),"x"),expected,rtol=3e-14)

    def test_conductivity_model_equation(self):
        e=np.array([.04,.1,.3])
        expected=.0018/(.020-1j*e)-1j*e*.0024/(.18**2-e**2-1j*.025*e)
        np.testing.assert_allclose(load_material("WTe2").sigma(to_omega(e,"eV"),"x"),expected,rtol=2e-14)

    def test_many_and_zero_oscillators(self):
        w=np.array([1e13,2e13])
        terms=tuple(Oscillator(1e26,1e14+i*1e12,1e12) for i in range(17))
        p=DrudeLorentzParameters(2,0,0,terms)
        expected=2+sum(t.amplitude/(t.omega_0**2-w*w-1j*t.gamma*w) for t in terms)
        np.testing.assert_allclose(evaluate_drude_lorentz(w,p),expected)
        np.testing.assert_allclose(evaluate_drude_lorentz(w,DrudeLorentzParameters(2,0,0,())),2)

    def test_singular_model(self):
        p=DrudeLorentzParameters(1,0,0,(Oscillator(1e26,1e13,0),))
        with self.assertRaises(MaterialError):evaluate_drude_lorentz(1e13,p)


class DefinitionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):self.temp.cleanup()

    def clone(self,name="DemoNK"):
        folder=self.root/name
        shutil.copytree(BUILTIN_DATABASE/name,folder)
        return folder

    def update(self,folder,**kwargs):
        path=folder/"material.yaml"
        meta=yaml.safe_load(path.read_text(encoding="utf-8"));meta.update(kwargs)
        path.write_text(yaml.safe_dump(meta),encoding="utf-8")
        return meta

    def rewrite(self,folder,transform):
        path=folder/"data.csv"
        rows=list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))));transform(rows)
        buf=io.StringIO();csv.writer(buf).writerows(rows);path.write_text(buf.getvalue(),encoding="utf-8")

    def test_missing_imaginary(self):
        folder=self.clone("DemoEpsilon")
        self.rewrite(folder,lambda rows:[r.pop() for r in rows])
        with self.assertRaisesRegex(MaterialError,"imaginary"):load_material(folder)

    def test_missing_point(self):
        folder=self.clone()
        self.rewrite(folder,lambda rows:rows[3].__setitem__(1,""))
        with self.assertRaisesRegex(MaterialError,"row 4"):load_material(folder)

    def test_nan_point(self):
        folder=self.clone()
        self.rewrite(folder,lambda rows:rows[2].__setitem__(2,"nan"))
        with self.assertRaisesRegex(MaterialError,"nonfinite"):load_material(folder)

    def test_duplicate_point(self):
        folder=self.clone()
        self.rewrite(folder,lambda rows:rows.append(rows[2].copy()))
        with self.assertRaisesRegex(MaterialError,"Duplicate spectral"):load_material(folder)

    def test_unsorted_rows_warn_and_sort(self):
        folder=self.clone()
        self.rewrite(folder,lambda rows:rows.__setitem__(slice(1,None),rows[:0:-1]))
        with self.assertWarns(MaterialWarning):mat=load_material(folder)
        self.assertTrue(mat.metadata["was_sorted"])
        self.assertTrue(np.all(np.diff(mat.omega)>0))
        np.testing.assert_allclose(mat.epsilon(to_omega(1.25,"eV")),load_material("DemoNK").epsilon(to_omega(1.25,"eV")))

    def test_negative_coordinate(self):
        folder=self.clone();self.rewrite(folder,lambda rows:rows[1].__setitem__(0,"-1"))
        with self.assertRaisesRegex(MaterialError,"positive"):load_material(folder)

    def test_hz_angular_header_conflict(self):
        folder=self.clone("DemoEpsilon");self.update(folder,spectral_units="rad/s")
        with self.assertRaisesRegex(MaterialError,"must not be mixed"):load_material(folder)

    def test_ambiguous_coordinate_columns(self):
        folder=self.clone()
        def transform(rows):
            rows[0].append("omega")
            for row in rows[1:]:row.append(row[0])
        self.rewrite(folder,transform)
        with self.assertRaisesRegex(MaterialError,"Multiple spectral"):load_material(folder)

    def test_unknown_units(self):
        folder=self.clone();self.update(folder,spectral_units="GHz")
        with self.assertRaisesRegex(MaterialError,"Unknown spectral"):load_material(folder)

    def test_metadata_wrong_type(self):
        folder=self.clone();self.update(folder,input_type=["nk"])
        with self.assertRaisesRegex(MaterialError,"input_type must"):load_material(folder)

    def test_column_mapping_wrong_type(self):
        folder=self.clone();self.update(folder,columns={"iso":{"n":["n"]}})
        with self.assertRaisesRegex(MaterialError,"column mapping"):load_material(folder)

    def test_negative_model_damping(self):
        folder=self.clone("Au");path=folder/"parameters.csv"
        path.write_text(path.read_text().replace("iso,gamma_d,,0.053,eV","iso,gamma_d,,-0.053,eV"),encoding="utf-8")
        with self.assertRaisesRegex(MaterialError,"nonnegative"):load_material(folder)

    def test_noninteger_oscillator_count(self):
        folder=self.clone("Au");path=folder/"parameters.csv"
        path.write_text(path.read_text().replace("iso,n_lorentz,,5,1","iso,n_lorentz,,5.5,1"),encoding="utf-8")
        with self.assertRaisesRegex(MaterialError,"nonnegative integer"):load_material(folder)

    def test_conductivity_dimensionality_mismatch(self):
        folder=self.clone("WTe2");self.update(folder,response_units="S/m")
        with self.assertRaisesRegex(MaterialError,"sheet or bulk"):load_material(folder)

    def test_missing_tensor_component(self):
        folder=self.clone("DemoBiaxial");self.update(folder,components=["x","y"])
        with self.assertRaisesRegex(MaterialError,"every principal axis"):load_material(folder)

    def test_missing_tensor_column(self):
        folder=self.clone("DemoBiaxial");self.rewrite(folder,lambda rows:[r.pop() for r in rows])
        with self.assertRaisesRegex(MaterialError,"epsilon_z_imag"):load_material(folder)

    def test_parameter_units_mismatch(self):
        folder=self.clone("Au");path=folder/"parameters.csv"
        path.write_text(path.read_text().replace(",eV\n",",Hz\n",1),encoding="utf-8")
        with self.assertRaisesRegex(MaterialError,"consistent spectral units"):load_material(folder)

    def test_duplicate_parameter(self):
        folder=self.clone("Au");path=folder/"parameters.csv";text=path.read_text()
        path.write_text(text+text.splitlines()[1]+"\n",encoding="utf-8")
        with self.assertRaisesRegex(MaterialError,"Duplicate parameter"):load_material(folder)

    def test_missing_parameter(self):
        folder=self.clone("Au");path=folder/"parameters.csv"
        path.write_text("\n".join(line for line in path.read_text().splitlines() if ",omega_p," not in line)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(MaterialError,"Missing parameter"):load_material(folder)

    def test_extra_parameter(self):
        folder=self.clone("Au");path=folder/"parameters.csv"
        path.write_text(path.read_text()+"iso,typo,,1,1\n",encoding="utf-8")
        with self.assertRaisesRegex(MaterialError,"unused"):load_material(folder)

    def test_metadata_range_exceeds_table(self):
        folder=self.clone();self.update(folder,valid_frequency_range={"min":.1,"max":3,"unit":"eV"})
        with self.assertRaisesRegex(MaterialError,"beyond the tabulated"):load_material(folder)

    def test_invalid_range(self):
        folder=self.clone();self.update(folder,valid_frequency_range={"min":3,"max":.5,"unit":"eV"})
        with self.assertRaisesRegex(MaterialError,"less than"):load_material(folder)

    def test_missing_metadata_field(self):
        folder=self.clone();path=folder/"material.yaml";meta=yaml.safe_load(path.read_text());del meta["time_convention"]
        path.write_text(yaml.safe_dump(meta),encoding="utf-8")
        with self.assertRaisesRegex(MaterialError,"time_convention"):load_material(folder)

    def test_unknown_source(self):
        folder=self.clone();self.update(folder,input_type="unknown")
        with self.assertRaisesRegex(MaterialError,"Unknown input_type"):load_material(folder)

    def test_path_escape(self):
        folder=self.clone();self.update(folder,data_file="../escape.csv")
        with self.assertRaisesRegex(MaterialError,"inside their own"):load_material(folder)

    def test_plus_convention_table(self):
        folder=self.clone("DemoEpsilon");self.update(folder,time_convention="exp(+i*omega*t)")
        def conjugate(rows):
            for row in rows[1:]:row[2]=str(-float(row[2]))
        self.rewrite(folder,conjugate)
        w=to_omega([1.,2.],"eV")
        np.testing.assert_allclose(load_material(folder).epsilon(w),load_material("DemoNK").epsilon(w),rtol=2e-14)

    def test_plus_convention_model(self):
        folder=self.clone("Au");self.update(folder,time_convention="exp(+i*omega*t)")
        w=to_omega([1.,2.],"eV")
        np.testing.assert_allclose(load_material(folder).epsilon(w),load_material("Au").epsilon(w))

    def test_tsv_json_whitespace(self):
        original=load_material("DemoNK")
        for extension,delimiter in [("tsv","\t"),("json",None),("txt","whitespace")]:
            with self.subTest(extension=extension):
                folder=self.root/extension;shutil.copytree(BUILTIN_DATABASE/"DemoNK",folder)
                records=list(csv.DictReader(io.StringIO((folder/"data.csv").read_text())))
                name="data."+extension
                if extension == "json":text=json.dumps([{k:float(v) for k,v in row.items()} for row in records])
                else:
                    separator=" " if delimiter=="whitespace" else delimiter
                    text=separator.join(records[0])+"\n"+"\n".join(separator.join(row.values()) for row in records)
                (folder/name).write_text(text,encoding="utf-8")
                options={"data_file":name}
                if delimiter == "whitespace":options["delimiter"]=delimiter
                self.update(folder,**options)
                np.testing.assert_allclose(load_material(folder).epsilon(to_omega(1.25,"eV")),original.epsilon(to_omega(1.25,"eV")))

    def test_custom_column_mapping(self):
        folder=self.clone();self.rewrite(folder,lambda rows:rows.__setitem__(0,["E","N","K"]))
        self.update(folder,spectral_column="E",columns={"iso":{"n":"N","kappa":"K"}})
        np.testing.assert_allclose(load_material(folder).epsilon(to_omega(1,"eV")),load_material("DemoNK").epsilon(to_omega(1,"eV")))

    def test_uniaxial_requires_equal_axes(self):
        folder=self.clone("DemoBiaxial");self.update(folder,symmetry="uniaxial")
        with self.assertRaisesRegex(MaterialError,"two equal"):load_material(folder)
        def equal_xy(rows):
            for row in rows[1:]:row[3:5]=row[1:3]
        self.rewrite(folder,equal_xy)
        self.assertEqual(load_material(folder).metadata["symmetry"],"uniaxial")

    def test_custom_adapter(self):
        folder=self.clone();self.update(folder,input_type="test_constant_adapter")
        def adapter(meta,payload):
            return dict(evaluator=lambda w,c:np.full(w.shape,2+.1j),native_response="epsilon",data_type="model",omega=None,parameters={},was_sorted=False)
        register_loader("test_constant_adapter",adapter)
        np.testing.assert_allclose(load_material(folder).epsilon(to_omega(1,"eV"),"x"),2+.1j)
        with self.assertRaises(MaterialError):register_loader("test_constant_adapter",adapter)
        save_material(load_material(folder),self.root/"adapter_saved")
        np.testing.assert_allclose(load_material(self.root/"adapter_saved").epsilon(to_omega(1,"eV"),"x"),2+.1j)

    def test_save_reload_all_preserves_source(self):
        for name in available_materials():
            with self.subTest(name=name):
                mat=load_material(name);saved=save_material(mat,self.root/("saved-"+name));restored=load_material(saved)
                w=np.geomspace(*mat.valid_frequency_range,11)
                np.testing.assert_array_equal(restored.sigma(w),mat.sigma(w))
                self.assertEqual(restored.metadata,mat.metadata)
                filename=mat.metadata.get("parameter_file",mat.metadata.get("data_file"))
                self.assertEqual((saved/filename).read_bytes(),(BUILTIN_DATABASE/name/filename).read_bytes())

    def test_save_refuses_overwrite(self):
        folder=self.clone()
        before=(folder/"data.csv").read_bytes()
        with self.assertRaises(MaterialError):save_material(load_material("Au"),folder)
        self.assertEqual(before,(folder/"data.csv").read_bytes())

    def test_new_folder_requires_no_code_changes(self):
        folder=self.root/"NewMaterial";shutil.copytree(BUILTIN_DATABASE/"DemoNK",folder)
        self.update(folder,name="NewMaterial",source_description="Local extension test")
        mat=load_material("NewMaterial",database=self.root)
        self.assertEqual(mat.name,"NewMaterial")
        self.assertIn("NewMaterial",available_materials(self.root))

    def test_unknown_material(self):
        with self.assertRaisesRegex(MaterialError,"was not found"):load_material("NoSuchMaterial")


class PlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_all_axes_without_mutation(self):
        mat=load_material("DemoNK");before=mat.omega
        for axis in ["eV","Hz","THz","cm-1","rad/s"]:
            fig,axes=mat.plot_epsilon(x_axis=axis,points=10)
            fig.canvas.draw()
            self.assertEqual(len(axes),2)
            self.assertEqual(len(axes[0].lines),1)
        np.testing.assert_array_equal(mat.omega,before)

    def test_anisotropic_comparable_axes(self):
        fig,axes=load_material("WTe2").plot_sigma(points=20)
        fig.canvas.draw()
        self.assertEqual(len(axes[0].lines),2)
        self.assertEqual(len(axes[1].lines),2)
        self.assertIn("illustrative",axes[0].get_title())

    def test_sheet_epsilon_plot_needs_thickness(self):
        with self.assertRaises(ResponseUnavailableError):load_material("WTe2").plot_epsilon()
        fig,axes=load_material("WTe2").plot_epsilon(thickness_m=1e-9,points=20)
        fig.canvas.draw();self.assertEqual(len(axes[0].lines),3)

    def test_bad_plot_grid_and_units(self):
        for options in [{"x_axis":"GHz"},{"points":1},{"omega":to_omega([[1,2]],"eV")}]:
            with self.subTest(options=options),self.assertRaises(MaterialError):load_material("Au").plot_epsilon(**options)


if __name__ == "__main__":unittest.main()
