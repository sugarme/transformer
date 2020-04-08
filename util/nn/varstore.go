package nn

import (
	"fmt"
	"log"
	"strings"
	"sync"

	ts "gorgonia.org/tensor"
)

// The separator is used to separate path elements in the tensor names.
const SEP string = "."

// Variables holds variable pointer.
// When variable store is frozen, trainable is still set to tree.
// However, the tensor is not set to require gradients.
type Variables struct {
	NamedVariables     map[string]ts.Tensor
	TrainableVariables []ts.Tensor
	Mut                sync.Mutex
}

/// VarStore is used to store variables used by one
// or multiple layers. It specifies a single device
// where all variables are stored.
type VarStore struct {
	Variables Variables
	device    Device
}

// Path is a variable store with an associated path
// for variable naming.
type Path struct {
	Path     []string
	VarStore VarStore
}

// Entry holds an entry corresponding to a give name in `Path`
type Entry struct {
	Name      string
	Variables Variables
	Path      Path
}

// NewVarStore create a new variable store on the specified device
func NewVarStore(device Device) VarStore {
	variables := Variables{
		NamedVariables:     make(map[string]ts.Tensor),
		TrainableVariables: []ts.Tensor{},
		Mut:                sync.Mutex{},
	}

	return VarStore{
		Variables: variables,
		device:    device,
	}
}

// Device get the device for this varstore
func (vs *VarStore) Device() Device {
	return vs.device
}

// Len returns number of tensors currently stored on this varstore
func (vs *VarStore) Len() uint {
	vs.Variables.Mut.Lock()
	variables := len(vs.Variables.NamedVariables)
	vs.Variables.Mut.Unlock()

	return uint(variables)
}

// IsEmpty checks whether there is no tensors are currently stored
// on this varstore
func (vs *VarStore) IsEmpty() bool {
	vs.Variables.Mut.Lock()
	defer vs.Variables.Mut.Unlock()

	if len(vs.Variables.NamedVariables) == 0 {
		return true
	}

	return false
}

// TrainableVariables returns all trainable variables for this varstore
func (vs *VarStore) TrainableVariables() []ts.Tensor {
	vs.Variables.Mut.Lock()
	defer vs.Variables.Mut.Unlock()

	variables := vs.Variables.TrainableVariables

	return variables
}

// GetVariables returns all variables currently stores in varstore
// along with their names.
func (vs *VarStore) GetVariables() map[string]ts.Tensor {
	vs.Variables.Mut.Lock()
	defer vs.Variables.Mut.Unlock()

	variables := vs.Variables.NamedVariables

	return variables
}

// Root returns the root path for this varstore
// Variables are named and organized using paths. This function
// returns the top level path for the varstore and can be combined
// with "/" to create sub-paths.
func (vs *VarStore) Root() Path {
	return Path{
		Path:     []string{},
		VarStore: *vs,
	}
}

// Save save the `varstore` variable values to a file
// Weight values for all the tensors currently stored in the `varstore`
// will be saved to a file.
func (vs *VarStore) Save(path string) error {
	vs.Variables.Mut.Lock()
	defer vs.Variables.Mut.Unlock()

	variables := vs.Variables.NamedVariables

	var namedTensors []ts.Tensor

	for _, t := range variables {
		namedTensors = append(namedTensors, t)
	}

	// TODO: save nameTensors to file
	return nil
}

// Load loads the `varstore` variable values from a file.
// weight values for all the tensors currently stored in
// the `varstore` gets loaded from the given file. The set
// of variables stored in the `varstore` is not changed, only
// the values for these tensors are modified.
func (vs *VarStore) Load(path string) error {
	// TODO: load multi with device
	/* let named_tensors = Tensor::load_multi_with_device(&path, self.device)?;
	 * let named_tensors: HashMap<_, _> = named_tensors.into_iter().collect();
	 * let mut variables = self.variables_.lock().unwrap();
	 * for (name, var) in variables.named_variables.iter_mut() {
	 *     match named_tensors.get(name) {
	 *         Some(src) => {
	 *             crate::no_grad(|| var.f_copy_(src).map_err(|e| format_err!("{}: {}", name, e)))?
	 *         }
	 *         None => return Err(format_err!("cannot find {} in {:?}", name, path.as_ref())),
	 *     }
	 * }
	 * Ok(()) */

	return nil
}

// LoadPartial loads the `varstore` variable values from a file if it exists.
// Weight values for the tensors currently stored in the `varstore` and the given
// file get loaded from the given file. If a variable in the var store is not present
// in the given file, it is skipped and its values are not updated. This method should
// be used if pre-trained weight for only parts of the model are available.
// The set of variables stored in the `varstore` is not changed, only the values
// for these tensors are modified.
func (vs *VarStore) LoadPartial(path string) error {
	// TODO: implement
	/* let named_tensors = Tensor::load_multi_with_device(&path, self.device)?;
	 * let named_tensors: HashMap<_, _> = named_tensors.into_iter().collect();
	 * let mut variables = self.variables_.lock().unwrap();
	 * let mut missing_variables = Vec::new();
	 * for (name, var) in variables.named_variables.iter_mut() {
	 *     match named_tensors.get(name) {
	 *         Some(src) => {
	 *             crate::no_grad(|| var.f_copy_(src).map_err(|e| format_err!("{}: {}", name, e)))?
	 *         }
	 *         None => {
	 *             missing_variables.push(name.to_owned());
	 *         }
	 *     }
	 * }
	 * Ok(missing_variables) */

	return nil
}

// Freeze freezes a `varstore`
// Gradients for the variables in this store are not tracked anymore.
func (vs *VarStore) Freeze() {
	vs.Variables.Mut.Lock()
	defer vs.Variables.Mut.Unlock()

	variables := vs.Variables.NamedVariables

	for _, v := range variables {
		// TODO:
		fmt.Println(v)
		// v.SetRequiresGrad(false)
	}
}

// Unfreeze unfreezes a `varstore`
// Gradients for the variables in this store are tracked again.
func (vs *VarStore) Unfreeze() {
	vs.Variables.Mut.Lock()
	defer vs.Variables.Mut.Unlock()

	variables := vs.Variables.NamedVariables

	for _, v := range variables {
		// TODO:
		fmt.Println(v)
		// v.SetRequiresGrad(true)
	}
}

// Copy copies variable values from a source `varstore` to this `varstore`
// All the variables in this `varstore` have to exist with the same name
// in the source `varstore`, otherwise return an error.
func (vs *VarStore) Copy(src VarStore) error {
	vs.Variables.Mut.Lock()
	defer vs.Variables.Mut.Unlock()
	// variables := vs.Variables

	src.Variables.Mut.Lock()
	defer src.Variables.Mut.Unlock()
	// srcVariables := src.Variables

	/*   device := vs.device
	 *
	 *   for name, _ := range variables.NamedVariables {
	 *     if ok, _ := srcVariables.NamedVariables[name]; !ok {
	 *       err := fmt.Errorf("cannot find %v in the source var store", name)
	 *       continue
	 *     }
	 *
	 *     srcVar = srcVariables.NamedVariables[name]
	 *     // TODO: copy to device
	 *     // crate::no_grad(|| var.f_copy_(&src_var.to_device(device)))?;
	 *
	 *   } */

	return nil

}

// Var creates a new variable
// The new variable is named according to the name parameter
// and has the specified shape. The variable is trainable, its
// gradient will be tracked. The variable uses a float tensor
// initialized as per the related argument.
func (p *Path) Var(name string, dims []int64, init InitT) ts.Tensor {
	v := Init(init, dims, p.VarStore.Device())

	return p.add(name, v, true)
}

func (p *Path) add(name string, tensor ts.Tensor, trainable bool) ts.Tensor {
	path := p.path(name)

	p.VarStore.Variables.Mut.Lock()
	defer p.VarStore.Variables.Mut.Unlock()

	variables := p.VarStore.Variables
	if _, ok := variables.NamedVariables[path]; ok {
		path = fmt.Sprintf("%v__%v", path, len(variables.NamedVariables))
	}

	if trainable {
		// TODO: Turn on `gradient tracking`
		// tensor.set_requires_grad(true)
		p.VarStore.Variables.TrainableVariables = append(p.VarStore.Variables.TrainableVariables, tensor)
	}

	variables.NamedVariables[path] = tensor

	return tensor

}

func (p *Path) path(name string) string {
	if strings.Contains(name, SEP) {
		log.Fatalf("variable name cannot contain %v %v", SEP, name)
	}

	if p.VarStore.IsEmpty() {
		return name
	}

	return fmt.Sprintf("%v%v%v", strings.Join(p.Path, SEP), SEP, name)
}

// TODO: continue... the below methods

/* //! Variable stores.
 *
 * /// An Entry holds an entry corresponding to a given name in Path.
 * #[derive(Debug)]

 * impl VarStore {
 *
 *     /// Saves the var-store variable values to a file.
 *     ///
 *     /// Weight values for all the tensors currently stored in the
 *     /// var-store gets saved in the given file.
 *     pub fn save<T: AsRef<std::path::Path>>(&self, path: T) -> Fallible<()> {
 *         let variables = self.variables_.lock().unwrap();
 *         let named_tensors = variables.named_variables.iter().collect::<Vec<_>>();
 *         Tensor::save_multi(named_tensors.as_slice(), path)
 *     }
 *
 *     /// Loads the var-store variable values from a file.
 *     ///
 *     /// Weight values for all the tensors currently stored in the
 *     /// var-store gets loaded from the given file. Note that the set of
 *     /// variables stored in the var-store is not changed, only the values
 *     /// for these tensors are modified.
 *     pub fn load<T: AsRef<std::path::Path>>(&mut self, path: T) -> Fallible<()> {
 *         let named_tensors = Tensor::load_multi_with_device(&path, self.device)?;
 *         let named_tensors: HashMap<_, _> = named_tensors.into_iter().collect();
 *         let mut variables = self.variables_.lock().unwrap();
 *         for (name, var) in variables.named_variables.iter_mut() {
 *             match named_tensors.get(name) {
 *                 Some(src) => {
 *                     crate::no_grad(|| var.f_copy_(src).map_err(|e| format_err!("{}: {}", name, e)))?
 *                 }
 *                 None => return Err(format_err!("cannot find {} in {:?}", name, path.as_ref())),
 *             }
 *         }
 *         Ok(())
 *     }
 *
 *     /// Loads the var-store variable values from a file if it exists.
 *     ///
 *     /// Weight values for the tensors currently stored in the var-store and the given file get
 *     /// loaded from the given file. If a variable in the var store is not present in the given file,
 *     /// it is skipped and its values are not updated. This method should be used if pre-trained
 *     /// weight for only parts of the model are available.
 *     /// Note that the set of variables stored in the var-store is not changed, only the values
 *     /// for these tensors are modified.
 *     ///
 *     /// Returns a String Vector containing the names of missing variables.
 *     pub fn load_partial<T: AsRef<std::path::Path>>(&mut self, path: T) -> Fallible<Vec<String>> {
 *         let named_tensors = Tensor::load_multi_with_device(&path, self.device)?;
 *         let named_tensors: HashMap<_, _> = named_tensors.into_iter().collect();
 *         let mut variables = self.variables_.lock().unwrap();
 *         let mut missing_variables = Vec::new();
 *         for (name, var) in variables.named_variables.iter_mut() {
 *             match named_tensors.get(name) {
 *                 Some(src) => {
 *                     crate::no_grad(|| var.f_copy_(src).map_err(|e| format_err!("{}: {}", name, e)))?
 *                 }
 *                 None => {
 *                     missing_variables.push(name.to_owned());
 *                 }
 *             }
 *         }
 *         Ok(missing_variables)
 *     }
 *
 *     /// Freezes a var store.
 *     ///
 *     /// Gradients for the variables in this store are not tracked
 *     /// anymore.
 *     pub fn freeze(&mut self) {
 *         let variables = self.variables_.lock().unwrap();
 *         for variable in variables.trainable_variables.iter() {
 *             let _v = variable.set_requires_grad(false);
 *         }
 *     }
 *
 *     /// Unfreezes a var store.
 *     ///
 *     /// Gradients for the variables in this store are tracked again.
 *     pub fn unfreeze(&mut self) {
 *         let variables = self.variables_.lock().unwrap();
 *         for variable in variables.trainable_variables.iter() {
 *             let _v = variable.set_requires_grad(true);
 *         }
 *     }
 *
 *     /// Copies variable values from a source var store to this var store.
 *     ///
 *     /// All the variables in this var store have to exist with the same
 *     /// name in the source var store, otherwise an error is returned.
 *     pub fn copy(&mut self, src: &VarStore) -> Fallible<()> {
 *         let mut variables = self.variables_.lock().unwrap();
 *         let src_variables = src.variables_.lock().unwrap();
 *         let device = self.device;
 *         for name in variables.named_variables.keys() {
 *             if !src_variables.named_variables.contains_key(name) {
 *                 bail!("cannot find {} in the source var store", name);
 *             }
 *         }
 *         for (name, var) in variables.named_variables.iter_mut() {
 *             let src_var = src_variables.named_variables.get(name).unwrap();
 *             crate::no_grad(|| var.f_copy_(&src_var.to_device(device)))?;
 *         }
 *         Ok(())
 *     }
 * }
 *
 * impl<'a> Path<'a> {
 *     /// Gets a sub-path of the given path.
 *     pub fn sub<T: std::string::ToString>(&'a self, s: T) -> Path<'a> {
 *         let s = s.to_string();
 *         if s.chars().any(|x| x == SEP) {
 *             panic!("sub name cannot contain {} {}", SEP, s);
 *         }
 *         let mut path = self.path.clone();
 *         path.push(s);
 *         Path {
 *             path,
 *             var_store: self.var_store,
 *         }
 *     }
 *
 *     /// Gets the device where the var-store variables are stored.
 *     pub fn device(&self) -> Device {
 *         self.var_store.device
 *     }
 *
 *
 *
 *     fn get_or_add_with_lock(
 *         &self,
 *         name: &str,
 *         tensor: Tensor,
 *         trainable: bool,
 *         mut variables: MutexGuard<Variables>,
 *     ) -> Tensor {
 *         let path = self.path(name);
 *         if let Some(var) = variables.named_variables.get(&path) {
 *             return var.shallow_clone();
 *         }
 *
 *         let tensor = if trainable {
 *             tensor.set_requires_grad(true)
 *         } else {
 *             tensor
 *         };
 *         if trainable {
 *             variables.trainable_variables.push(tensor.shallow_clone());
 *         }
 *         variables
 *             .named_variables
 *             .insert(path, tensor.shallow_clone());
 *         tensor
 *     }
 *
 *     /// Creates a new variable initialized with zeros.
 *     ///
 *     /// The new variable is named according to the name parameter and
 *     /// has the specified shape. The variable will not be trainable so
 *     /// gradients will not be tracked.
 *     /// The variable uses a float tensor initialized with zeros.
 *     pub fn zeros_no_train(&self, name: &str, dims: &[i64]) -> Tensor {
 *         let z = Tensor::zeros(dims, (Kind::Float, self.device()));
 *         self.add(name, z, false)
 *     }
 *
 *     /// Creates a new variable initialized with ones.
 *     ///
 *     /// The new variable is named according to the name parameter and
 *     /// has the specified shape. The variable will not be trainable so
 *     /// gradients will not be tracked.
 *     /// The variable uses a float tensor initialized with ones.
 *     pub fn ones_no_train(&self, name: &str, dims: &[i64]) -> Tensor {
 *         let o = Tensor::ones(dims, (Kind::Float, self.device()));
 *         self.add(name, o, false)
 *     }
 *
 *     /// Creates a new variable initialized with zeros.
 *     ///
 *     /// The new variable is named according to the name parameter and
 *     /// has the specified shape. The variable is trainable, its gradient
 *     /// will be tracked.
 *     /// The variable uses a float tensor initialized with zeros.
 *     pub fn zeros(&self, name: &str, dims: &[i64]) -> Tensor {
 *         self.var(name, dims, Init::Const(0.))
 *     }
 *
 *     /// Creates a new variable initialized with ones.
 *     ///
 *     /// The new variable is named according to the name parameter and
 *     /// has the specified shape. The variable is trainable, its gradient
 *     /// will be tracked.
 *     /// The variable uses a float tensor initialized with ones.
 *     pub fn ones(&self, name: &str, dims: &[i64]) -> Tensor {
 *         self.var(name, dims, Init::Const(1.))
 *     }
 *
 *     /// Creates a new variable initialized randomly with normal distribution.
 *     ///
 *     /// The new variable is named according to the name parameter and
 *     /// has the specified shape. The variable is trainable, its gradient
 *     /// will be tracked.
 *     /// The variable uses a float tensor initialized randomly using a
 *     /// standard normal distribution.
 *     pub fn randn_standard(&self, name: &str, dims: &[i64]) -> Tensor {
 *         let init = Init::Randn {
 *             mean: 0.,
 *             stdev: 1.,
 *         };
 *         self.var(name, dims, init)
 *     }
 *
 *     /// Creates a new variable initialized randomly with normal distribution.
 *     ///
 *     /// The new variable is named according to the name parameter and
 *     /// has the specified shape. The variable is trainable, its gradient
 *     /// will be tracked.
 *     /// The variable uses a float tensor initialized randomly using a
 *     /// normal distribution with the specified mean and standard deviation.
 *     pub fn randn(&self, name: &str, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
 *         self.var(name, dims, Init::Randn { mean, stdev })
 *     }
 *
 *     /// Creates a new variable initialized randomly with uniform distribution.
 *     ///
 *     /// The new variable is named according to the name parameter and
 *     /// has the specified shape. The variable is trainable, its gradient
 *     /// will be tracked.
 *     /// The variable uses a float tensor initialized randomly using a
 *     /// uniform distribution between the specified bounds.
 *     pub fn uniform(&self, name: &str, dims: &[i64], lo: f64, up: f64) -> Tensor {
 *         self.var(name, dims, Init::Uniform { lo, up })
 *     }
 *
 *     /// Creates a new variable initialized randomly with kaiming uniform.
 *     ///
 *     /// The new variable is named according to the name parameter and
 *     /// has the specified shape. The variable is trainable, its gradient
 *     /// will be tracked.
 *     /// The variable uses a float tensor initialized randomly using a
 *     /// uniform distribution which bounds follow Kaiming initialization.
 *     pub fn kaiming_uniform(&self, name: &str, dims: &[i64]) -> Tensor {
 *         self.var(name, dims, Init::KaimingUniform)
 *     }
 *
 *     /// Creates a new variable initialized by copying an existing tensor.
 *     ///
 *     /// The new variable is named according to the name parameter and
 *     /// has the specified shape. The variable is trainable, its gradient
 *     /// will be tracked.
 *     /// The variable uses a float tensor initialized by copying some
 *     /// given tensor.
 *     pub fn var_copy(&self, name: &str, t: &Tensor) -> Tensor {
 *         let mut v = self.zeros(name, &t.size());
 *         crate::no_grad(|| v.copy_(&t));
 *         v
 *     }
 *
 *     /// Gets the tensor corresponding to a given name if present.
 *     pub fn get(&self, name: &str) -> Option<Tensor> {
 *         let path = self.path(name);
 *         let variables = self.var_store.variables_.lock().unwrap();
 *         variables
 *             .named_variables
 *             .get(&path)
 *             .map(|v| v.shallow_clone())
 *     }
 *
 *     /// Gets the entry corresponding to a given name for in-place manipulation.
 *     pub fn entry<'b>(&'b self, name: &'b str) -> Entry<'b> {
 *         let variables = self.var_store.variables_.lock().unwrap();
 *         Entry {
 *             name,
 *             variables,
 *             path: &self,
 *         }
 *     }
 * }
 *
 * impl<'a> Entry<'a> {
 *     /// Returns the existing entry if, otherwise create a new variable.
 *     ///
 *     /// If this entry name matches the name of a variables stored in the
 *     /// var store, the corresponding tensor is returned. Otherwise a new
 *     /// variable is added to the var-store with the entry name and is
 *     /// initialized according to the init parameter.
 *     pub fn or_var(self, dims: &[i64], init: Init) -> Tensor {
 *         let v = super::init(init, dims, self.path.device());
 *         self.path
 *             .get_or_add_with_lock(self.name, v, true, self.variables)
 *     }
 *
 *     /// Returns the existing entry if, otherwise create a new variable.
 *     pub fn or_var_copy(self, tensor: &Tensor) -> Tensor {
 *         let mut v = self.or_zeros(&tensor.size());
 *         crate::no_grad(|| v.copy_(&tensor));
 *         v
 *     }
 *
 *     /// Returns the existing entry if, otherwise create a new variable.
 *     pub fn or_kaiming_uniform(self, dims: &[i64]) -> Tensor {
 *         self.or_var(dims, Init::KaimingUniform)
 *     }
 *
 *     /// Returns the existing entry if, otherwise create a new variable.
 *     pub fn or_ones(self, dims: &[i64]) -> Tensor {
 *         self.or_var(dims, Init::Const(1.))
 *     }
 *
 *     /// Returns the existing entry if, otherwise create a new variable.
 *     pub fn or_ones_no_train(self, dims: &[i64]) -> Tensor {
 *         let o = Tensor::ones(dims, (Kind::Float, self.path.device()));
 *         self.path
 *             .get_or_add_with_lock(self.name, o, true, self.variables)
 *     }
 *
 *     /// Returns the existing entry if, otherwise create a new variable.
 *     pub fn or_randn(self, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
 *         self.or_var(dims, Init::Randn { mean, stdev })
 *     }
 *
 *     /// Returns the existing entry if, otherwise create a new variable.
 *     pub fn or_randn_standard(self, dims: &[i64]) -> Tensor {
 *         let init = Init::Randn {
 *             mean: 0.,
 *             stdev: 1.,
 *         };
 *         self.or_var(dims, init)
 *     }
 *
 *     /// Returns the existing entry if, otherwise create a new variable.
 *     pub fn or_uniform(self, dims: &[i64], lo: f64, up: f64) -> Tensor {
 *         self.or_var(dims, Init::Uniform { lo, up })
 *     }
 *
 *     /// Returns the existing entry if, otherwise create a new variable.
 *     pub fn or_zeros(self, dims: &[i64]) -> Tensor {
 *         self.or_var(dims, Init::Const(0.))
 *     }
 *
 *     /// Returns the existing entry if, otherwise create a new variable.
 *     pub fn or_zeros_no_train(self, dims: &[i64]) -> Tensor {
 *         let z = Tensor::zeros(dims, (Kind::Float, self.path.device()));
 *         self.path
 *             .get_or_add_with_lock(self.name, z, true, self.variables)
 *     }
 * }
 *
 * impl<'a, T> Div<T> for &'a mut Path<'a>
 * where
 *     T: std::string::ToString,
 * {
 *     type Output = Path<'a>;
 *
 *     fn div(self, rhs: T) -> Self::Output {
 *         self.sub(rhs.to_string())
 *     }
 * }
 *
 * impl<'a, T> Div<T> for &'a Path<'a>
 * where
 *     T: std::string::ToString,
 * {
 *     type Output = Path<'a>;
 *
 *     fn div(self, rhs: T) -> Self::Output {
 *         self.sub(rhs.to_string())
 *     }
 * } */
