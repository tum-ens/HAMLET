Machine Learning Forecasting Methods
===============================

Introduction
-----------

Machine learning forecasting methods leverage advanced algorithms to learn patterns from historical data and make predictions about future values. These methods can capture complex, non-linear relationships and incorporate multiple input features, making them particularly effective for forecasting energy-related time series that depend on various factors such as weather, occupancy patterns, and seasonal effects.

In the spectrum of forecasting complexity, machine learning methods represent the more sophisticated end, offering potentially higher accuracy at the cost of increased computational requirements and reduced interpretability compared to simpler methods.

Random Forest Regressor (RFR)
-----------------------------

Random Forest Regressor is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the average prediction of the individual trees. This method is robust to overfitting and can capture non-linear relationships between input features and the target variable.

Available for
~~~~~~~~~~~~

The RFR method is available for most components in HAMLET:

Inflexible Load | Heat Demand | DHW

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(\mathbf{x})

where:

   - :math:`\hat{y}` is the forecasted value
   - :math:`B` is the number of trees in the forest
   - :math:`f_b(\mathbf{x})` is the prediction of the :math:`b`-th tree
   - :math:`\mathbf{x}` is the input feature vector (e.g., time, temperature, etc.)

Configuration
~~~~~~~~~~~~

RFR forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   inflexible-load:
     fcast:
       method: rfr  # Options: perfect, naive, average, smoothed, sarma, rfr, cnn, rnn, arima
       rfr:
         features: ['temp', 'time']  # used features in weather file to fit the model
         days: 3  # past days that are used to train the model

Convolutional Neural Network (CNN)
---------------------------------

Convolutional Neural Networks are deep learning models that use convolutional layers to automatically and adaptively learn spatial hierarchies of features from input data. In time series forecasting, CNNs can capture local patterns and temporal dependencies in the data.

Available for
~~~~~~~~~~~~

The CNN method is available for most components in HAMLET:

Inflexible Load | Heat Demand | DHW

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \hat{y} = f_{\text{out}}(f_L(...f_2(f_1(\mathbf{X}; \mathbf{W}_1); \mathbf{W}_2)...; \mathbf{W}_L); \mathbf{W}_{\text{out}})

where:

   - :math:`\hat{y}` is the forecasted value
   - :math:`\mathbf{X}` is the input data (e.g., time series and features)
   - :math:`f_l` is the function of the :math:`l`-th convolutional layer
   - :math:`\mathbf{W}_l` are the weights of the :math:`l`-th layer
   - :math:`L` is the number of convolutional layers
   - :math:`f_{\text{out}}` is the output layer function
   - :math:`\mathbf{W}_{\text{out}}` are the weights of the output layer

Configuration
~~~~~~~~~~~~

CNN forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   inflexible-load:
     fcast:
       method: cnn  # Options: perfect, naive, average, smoothed, sarma, rfr, cnn, rnn, arima
       cnn:
         features: ['temp', 'time']  # used features in weather file to fit the model
         days: 3  # past days that are used to train the neural network
         epoch: 20  # number of epochs to fit the neural network model
         window_length: 20  # window length of the training data

Recurrent Neural Network (RNN)
----------------------------

Recurrent Neural Networks are a class of neural networks designed for sequential data processing. They maintain an internal state (memory) that allows them to capture temporal dependencies in time series data, making them well-suited for forecasting tasks.

Available for
~~~~~~~~~~~~

The RNN method is available for most components in HAMLET:

Inflexible Load | Heat Demand | DHW

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \mathbf{h}_t = f_h(\mathbf{W}_{hx}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h)

   \hat{y}_t = f_y(\mathbf{W}_{yh}\mathbf{h}_t + \mathbf{b}_y)

where:

   - :math:`\mathbf{x}_t` is the input at time step :math:`t`
   - :math:`\mathbf{h}_t` is the hidden state at time step :math:`t`
   - :math:`\hat{y}_t` is the output (forecast) at time step :math:`t`
   - :math:`\mathbf{W}_{hx}`, :math:`\mathbf{W}_{hh}`, :math:`\mathbf{W}_{yh}` are weight matrices
   - :math:`\mathbf{b}_h`, :math:`\mathbf{b}_y` are bias vectors
   - :math:`f_h`, :math:`f_y` are activation functions

Configuration
~~~~~~~~~~~~

RNN forecasting methods can be configured in the agent configuration file:

.. code-block:: yaml

   inflexible-load:
     fcast:
       method: rnn  # Options: perfect, naive, average, smoothed, sarma, rfr, cnn, rnn, arima
       rnn:
         features: ['temp', 'time']  # used features in weather file to fit the model
         days: 3  # past days that are used to train the neural network
         epoch: 20  # number of epochs to fit the neural network model
         window_length: 20  # window length of the training data

Notes
~~~~~

Machine learning forecasting methods in HAMLET have the following characteristics:

1. **Feature Engineering**:
   - All ML methods can incorporate external features like temperature and time
   - Features must be available in the weather file or derived from time information
   - Feature selection can significantly impact forecast accuracy

2. **Training Process**:
   - Models are trained using historical data from the specified number of days
   - Training occurs periodically based on the retraining parameter
   - For neural networks (CNN, RNN), training involves multiple epochs

3. **Computational Requirements**:
   - ML methods are more computationally intensive than simpler methods
   - Training can be time-consuming, especially for neural networks
   - Inference (generating forecasts) is relatively fast once models are trained

4. **Implementation Details**:
   - HAMLET uses scikit-learn for Random Forest Regressor
   - TensorFlow/Keras is used for CNN and RNN implementations
   - Models are saved and loaded to avoid retraining at every timestep
   - Window-based approaches are used to prepare training data

5. **Advantages and Limitations**:
   - **Advantages**:
     - Can capture complex, non-linear relationships
     - Ability to incorporate multiple features
     - Potentially higher accuracy for complex patterns
     - Adaptability to different types of data

   - **Limitations**:
     - Require more historical data for training
     - More computationally intensive
     - Less interpretable than simpler methods
     - Risk of overfitting, especially with limited data
     - Require careful hyperparameter tuning

6. **Practical Considerations**:
   - Start with simpler methods before moving to ML approaches
   - Ensure sufficient historical data is available
   - Consider computational resources, especially for large-scale simulations
   - Validate models carefully to avoid overfitting
   - Balance accuracy gains against increased complexity