# Migrate to Model Card Toolkit 3

Model Card Toolkit 3 is designed to be model framework agnostic. Unlike in previous
versions, you now need to install the extra package `model-card-toolkit[tensorflow]`
to use Model Card Toolkit's TensorFlow utilities.

We've tried to balance minimizing breaking changes and having a simple and
consistent API. The following table lists the backwards incompatible changes
to the API. The primary change is that utilities that require optional TensorFlow
dependencies are in files prefixed with `tf_`.

<table>
  <tr>
    <th>Removed</th>
    <th>Replacement</th> 
  </tr>
  <tr>
    <td>TensorFlow utils in model_card_toolkit.utils.graphics</td> 
    <td>model_card_toolkit.utils.tf_graphics</td>
  </tr>
  <tr>
    <td>model_card_toolkit.utils.json_util</td> 
    <td>model_card_toolkit.utils.json_utils</td>
  </tr>
  <tr>
    <td>model_card_toolkit.utils.testdata.testdata_utils</td> 
    <td>model_card_toolkit.utils.testdata.tf_testdata_utils</td>
  </tr>
  <tr>
    <td>model_card_toolkit.utils.sources</td> 
    <td>model_card_toolkit.utils.tf_sources</td>
  </tr>
  <tr>
    <td>model_card_toolkit.utils.tfx_utils</td> 
    <td>model_card_toolkit.utils.tf_utils</td>
  </tr>
  <tr>
    <td>model_card_toolkit.utils.validation</td> 
    <td>model_card_toolkit.utils.json_utils</td>
  </tr>
</table>
