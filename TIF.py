from Utils import check_types, normalize_signal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.signal import butter, filtfilt, iirnotch, spectrogram, hilbert, find_peaks, welch
from scipy.fft import fft, fftfreq

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Info():
    """ Clase para almacenar información acerca del registro de datos.
       Esta clase se comporta como un diccionario. """

    def __init__(self, ch_names:list= [], ch_types:list= [], bads:list= [], sfreq:float= 512, description:str= None, experimenter:str= "No data", subject_info:dict= None):
        """Genera un objeto Info().

        Parameters:
            ch_names (list of str, optional): Lista con los nombres de los canales (por defecto es una lista vacia []).   
            ch_types (str or list of str, optional): Tipo de cada canal (ej: "eeg", "ecg", "emg") o un único tipo para todos (por defecto es una lista vacia []).  
            bads (list of str, optional): Lista de canales marcados como “malos” (por defecto es una lista vacia []).  
            sfreq (float, optional): Frecuencia de muestreo en Hz (por defecto es 512 Hz).  
            description (str, optional): Descripción sobre el registro de datos. 
            experimenter (str, optional): Nombre del experimentador (por defecto es "No data").  
            subject_info (dict, optional): Información adicional del sujeto.

        Raises:
            ValueError: Si "ch_names" y "ch_types" son listas de distinta longitud.
            TypeError: Si algún valor no es del tipo esperado."""

        ########## Validación de tipos ##########
        
        check_types( (ch_names, list), (ch_types, (str, list)), (bads, list), (sfreq, (int, float)), (description, (str, type(None))), (experimenter, str), (subject_info, (dict, type(None))) )

        ########## Validación de canales duplicados en "ch_names" ##########
        
        if isinstance(ch_names, list):
            if len(ch_names) != len(set(ch_names)):
                raise ValueError("Error: No puede haber canales duplicados en 'ch_name'.")
        
        ########## Validación de cantidad de elementos entre "ch_names" y "ch_types" ##########
        
        if isinstance(ch_types, list) and isinstance(ch_names, list):
            if len(ch_types) != len(ch_names):
                raise ValueError("Error: Si 'ch_types' es lista, debe tener la misma longitud que 'ch_name'.")
        
        ########## Validación de la frecuencia de muestreo ##########
        
        if sfreq <= 0:
            raise ValueError(f"Error: La frecuencia de muestreo '{sfreq}' no puede ser igual o menor a cero.")
        
        ########## Validación de los canales marcados como "malos" ##########
        
        if isinstance(ch_names, list):
            
            if len(bads) != len(set(bads)):
                raise ValueError("Error: No puede haber canales duplicados en 'bads'.")
            
            if len(ch_names) > 0 and len(bads) == len(ch_names):
                raise ValueError("Error: Todos los canales no pueden estar marcados como malos.")
            
            invalid_bads = list(set(bads) - set(ch_names))
            
            if invalid_bads != []:
                raise ValueError(f"Error: Hay canales en 'bads' que no están en 'ch_names': {invalid_bads}")        
        
        ########## Creación del diccionario ##########
        
        self._data = {
            "ch_names": ch_names,
            "ch_types": [ch_types] * len(ch_names) if isinstance(ch_types, str) else ch_types,
            "bads": bads,
            "sfreq": sfreq,
            "description": description,
            "experimenter": experimenter,
            "subject_info": subject_info
                     }
        
    @property
    def ch_names(self):
        """Devuelve los nombres de los canales."""
        return self._data["ch_names"]
    
    @property
    def ch_types(self):
        """Devuelve los tipos de los canales."""
        return self._data["ch_types"]
    
    @property
    def bads(self):
        """Devuelve los nombres de los canales marcados como 'malos'."""
        return self._data["bads"]
    
    @property
    def sfreq(self):
        """Devuelve la frecuencia de muestreo."""
        return self._data["sfreq"]
    
    @property
    def description(self):
        """Devuelve la descripción del experimento."""
        return self._data["description"]
    
    @property
    def experimenter(self):
        """Devuelve el nombre del experimentador."""
        return self._data["experimenter"]
    
    @property
    def subject_info(self):
        """Devuelve información sobre el sujeto experimentado."""
        return self._data["subject_info"]

    def __contains__(self, key:str) -> bool: 
        """Permite verificar si una clave está presente en el objeto.
        
        Parameters:
            key (str): Clave a verificar existencia."""
        
        return True if key in self._data else False

    def __getitem__(self, key:str) -> str:
        """Permite acceder a elementos como si fuera un diccionario.
        
        Parameters:
            key (str): Clave a acceder.
            
        Raises:
            KeyError: Si la clave no existe en el objeto."""
        
        try:
            return self._data[key]
        
        except KeyError:
            raise KeyError(f"La clave '{key}' no existe en Info.")

    def __len__(self) -> int:
        """Devuelve la cantidad de elementos almacenados."""
        
        return len(self._data)

    def get(self, key:str):
        """Obtiene el valor de una clave específica.
        
        Parameters:
            key (str): Clave a acceder."""

        return self._data.get(key)

    def keys(self):
        """Devuelve un iterador con las claves del objeto."""
        
        return self._data.keys()

    def items(self):
        """Devuelve un iterador con los pares (clave, valor) almacenados."""
        
        return self._data.items()

    def rename_channels(self, channel_renames:dict) -> None:
        """Permite renombrar canales de forma segura.

        Parameters:
            channel_renames (dict): Diccionario con los "nombres antiguos de canales" como "claves" y los "nombres nuevos de canales" como "valores".

        Raises:
            ValueError: Si alguna clave de channel_renames no existe en "ch_names", o si los nuevos nombres introducen duplicados."""

        check_types( (channel_renames, dict) )
        
        if self._data["ch_names"] and self._data["bads"] is not None:
            
            if not all(isinstance(new_name, str) for new_name in channel_renames.values()):
                raise ValueError(f"Error: Los nuevos nombres para los canales deben ser 'str'.")

            for old_channel in channel_renames:
                if old_channel not in self._data["ch_names"]:
                    raise ValueError(f"No existe un canal con el nombre '{old_channel}' en 'ch_names'.")
        
            new_ch_names = [channel_renames.get(name, name) for name in self._data["ch_names"]]
            
            if len(new_ch_names) != len(set(new_ch_names)):
                raise ValueError("El renombrado generaría nombres duplicados en 'ch_names'.")

            new_bads = [channel_renames.get(bad, bad) for bad in self._data["bads"]]

            self._data["ch_names"] = new_ch_names
            self._data["bads"] = new_bads
            
class Anotaciones:
    """ Clase para almacenar y gestionar anotaciones (eventos fisiológicos) en registros fisiológicos.
        Permite la adición, eliminación, modificación y búsqueda en CSV. """

    def __init__(self, onset:list, duration:list= None, description:list= None):
        """Inicializa la clase con los datos de las anotaciones.

        Parameters:
            onset (list of float): Lista de tiempos de inicio de eventos fisiológicos (en segundos).
            duration (list of float, optional): Lista de duraciones de eventos fisiológicos (en segundos).
            description (list of str, optional): Lista de descripciones de eventos fisiológicos.

        Raises:
            ValueError: Si las listas "onset", "duration" y "description" tienen longitudes distintas o falta alguna de ellas cuando se pasa "onset"."""
        
        ########## Validación de tipos ##########
        
        check_types( (onset, list), (duration, (list, type(None))), (description, (list, type(None))) )
        
        ########## Validación de tiempos no negativos ##########
        
        if any(time < 0 for time in onset):
            raise ValueError("Todos los valores en 'onset' deben ser mayores o iguales a 0.")

        if duration is not None:
            
            for time in duration:
                if time is not None and time <= 0:
                    raise ValueError("Todos los valores en 'duration' deben ser mayores que 0.")
        
        ########## Validación de orden en onset ##########
        
        if any(onset[i] > onset[i + 1] for i in range(len(onset) - 1)):
            raise ValueError("Los valores de 'onset' deben estar en orden creciente (de menor a mayor).")
        
        self.onset = onset

        ########## Caso: No se proporciona la lista "duration" ##########
        
        if duration is None:
            self.duration = [np.nan] * len(self.onset)
        
        else:
            self.duration = [np.nan if time is None else time for time in duration]  # Se rellena valores no conocidos (None) con NaN
            
        ########## Caso: No se proporciona la lista "description" ##########
        
        if description is None:
            self.description = ["No data"] * len(self.onset)
        
        else:
            self.description = description
        
        ########## Caso: Las listas "duration" / "description" no tienen la misma cantidad de elementos que "onset" ##########
        
        if len(self.duration) != len(self.onset):
            raise ValueError("La lista 'duration' debe tener la misma longitud que 'onset'.")

        if len(self.description) != len(self.onset):
            raise ValueError("La lista 'description' debe tener la misma longitud que 'onset'.")
                    
        self.df_eventos = pd.DataFrame({
                                        "onset": self.onset,
                                        "duration": self.duration,
                                        "description": self.description
                                       })

    @classmethod
    def load(cls, filepath: str, sfreq: float = 512) -> "Anotaciones":
        """Carga anotaciones desde un archivo CSV.
           
        Parameters:
            filepath (str): Ruta del archivo CSV con las anotaciones.
            fs (float, optional): Frecuencia de muestreo en Hz. (por defecto es 512)
        
        Returns:
            Anotaciones: Instancia de la clase con los datos convertidos."""
        
        check_types((filepath, str), (sfreq, (float, int, type(None))))
        
        loaded_df = pd.read_csv(filepath)

        if "onset" not in loaded_df.columns or "duration" not in loaded_df.columns:
            raise ValueError("El archivo CSV debe contener las columnas 'onset' y 'duration'.")

        # Aceptar "description" o "event_id" como nombre de columna

        if "description" in loaded_df.columns:
            desc_col = "description"
        
        elif "event_id" in loaded_df.columns:
            desc_col = "event_id"
        
        else:
            raise ValueError("El archivo CSV debe contener la columna 'description' o 'event_id'.")

        onsets = loaded_df["onset"].tolist()

        # Comprobar si la columna "onset" son muestras
        
        all_sample = all(float(x).is_integer() for x in onsets)
        
        mean_sample = np.mean(onsets)

        if all_sample and mean_sample > 1000:
            onsets = [sample / sfreq for sample in onsets]

        return Anotaciones(
            onset=onsets,
            duration= loaded_df["duration"].tolist(),
            description= loaded_df[desc_col].tolist()
                          )
        
    def save(self, filepath: str) -> None:
        """Guarda las anotaciones en un archivo CSV.
        
        Parameters:
            filepath (str): Ruta del archivo CSV con las anotaciones."""
        
        check_types((filepath, str))
        
        self.df_eventos.to_csv(filepath, index= False)
    
    def add(self, new_onset:float, new_duration:float, new_description:str) -> None:
        """Agrega una nueva anotación.

        Parameters:
            new_onset (float): Tiempo de inicio del evento (en segundos).
            new_duration (float): Duración del evento (en segundos), puede ser None si no se conoce.
            new_description (str): Descripción del evento."""
        
        check_types( (new_onset, float), (new_duration, (float, type(None))), (new_description, str) )
        
        ########## Validar que "new_onset" sea mayor al último "onset" ##########
        if new_onset <= self.onset[-1]:
            raise ValueError(f"El valor de 'onset' ({new_onset}) debe ser mayor al último existente ({self.onset[-1]}).")

        ########## Validar que la descripción no esté repetida ##########
        if new_description in self.description:
            raise ValueError(f"La descripción '{new_description}' ya existe. No se permiten duplicados.")
        
        nueva = pd.DataFrame( [{
                        "onset": new_onset,
                        "duration": np.nan if new_duration is None else new_duration,
                        "description": new_description
                              }] ) 
        
        self.df_eventos = pd.concat([self.df_eventos, nueva], axis= 0, ignore_index= True)
        
        self.onset.append(new_onset)
        self.duration.append(new_duration)
        self.description.append(new_description)

    def remove(self, index:int = None, description:str = None) -> None:
        """Elimina una anotación por índice o descripción exacta.
        
        Parameters:
            index (int): Índice del evento.
            description (str): Descripción del evento.
        
        Raises:
            ValueError: Si el índice no existe, o la descripción no se encuentra."""
            
        check_types( (index, (int, type(None))), (description, (str, type(None))) )
    
        if index is not None:
            
            if index in self.df_eventos.index:
                self.df_eventos.drop(index, inplace= True)
            
            else:
                raise ValueError(f"Índice {index} no existe en el DataFrame de los eventos/anotaciones")
        
        elif description is not None:

            if description not in self.df_eventos["description"].values:
                raise ValueError(f"No se encontró ningún evento/anotación con descripción '{description}'")

            self.df_eventos = self.df_eventos[self.df_eventos["description"] != description]
        
        else:
            raise ValueError("Debe especificar index o description para eliminar")

        self.df_eventos.reset_index(drop= True, inplace= True)
        
        self.onset = self.df_eventos["onset"].tolist()
        self.duration = self.df_eventos["duration"].tolist()
        self.description = self.df_eventos["description"].tolist()

    def get_annotations(self) -> pd.DataFrame:
        """Devuelve un DataFrame con todas las anotaciones."""
        
        return self.df_eventos.copy(deep= True)   # Se retorna una copia para evitar que modificaciones externas alteren el DataFrame original

    def find(self, description:str = None) -> pd.DataFrame:
        """Busca y devuelve las anotaciones que coincidan con una descripción específica.
        
        Parameters:
            description (str): Descripción del evento. Si es None, devuelve los eventos/anotaciones sin descripción conocida ("No Data").
        
        Returns:
            DataFrame: Una copia del DataFrame original con las filas coincidentes. Si no hay coincidencias, devuelve un DataFrame vacío con encabezado."""
        
        if description is None:
            filtro = "No data"
        
        else:
            check_types((description, str))
            filtro = description

        coincidencias = self.df_eventos[self.df_eventos["description"] == filtro]

        if coincidencias.empty == True:
            raise ValueError(f"Aviso: No se encontró ningún evento/anotación con descripción '{filtro}'.")

        return coincidencias.copy(deep=True)

class RawSignal:
    """ Clase para manejar señales fisiológicas en formato NumPy.
        Este constructor permite inicializar el objeto "RawSignal" a partir de un array de datos, con información adicional de los canales y el índice de la primera muestra. """

    def __init__(self, data:np.ndarray, sfreq:int, info:Info= None, first_samp:int= 0, anotaciones:Anotaciones= None):
        """Inicializa una instancia de la clase RawSignal.
        
        Parameters:
            data (np.ndarray): Matriz de datos con forma "(n_canales , n_muestras)" que almacena las señales fisiológicas.
            sfreq (int): Frecuencia de muestreo de la señal en Hz.
            info (Info object, optional): Objeto del tipo Info que almacena información sobre el registro de la señal, contiene los nombres y tipos de canales, canales "malos", la frecuencia de muestreo e informacion sobre el sujeto, el experimento y el experimentador.
            first_samp (int, optional): Índice (muestra) del primer muestreo a utilizar de la señal (default es 0).
            anotaciones (Anotaciones object, optional): Objeto del tipo Anotaciones que almacena los eventos asociados a la señal y al experimento (inicio y duración en que sucedió un estimulo y su descripción).
        
        Raises:
            ValueError: Si el array "data" no tiene la forma "(n_canales , n_muestras)".
            ValueError: Si el índice "first_samp" está fuera del rango de la señal."""
        
        ########## Validación de tipos ##########
        
        check_types( (data, np.ndarray), (sfreq, int), (info, (Info, type(None))), (first_samp, int), (anotaciones, (Anotaciones, type(None))) )
        
        ########## Validación de "data" ##########
        
        if data.ndim == 1:
            data = data.reshape(1, -1)  # Convierte a (1, n_muestras)
        
        elif data.ndim != 2:
            raise ValueError("El array 'data' debe ser una matriz de 1 o 2 dimensiones.")
        
        if not np.issubdtype(data.dtype, np.floating): # Se utiliza para comprobar el tipo de matrices
            raise ValueError("El array 'data' debe contener valores del tipo flotante.")
        
        self.data = data.copy() # Para poder modificarlo sin cambiar la señal original
        self.n_channels, self.n_samples = self.data.shape
        
        ########## Validación de la "sfreq" ##########
        
        if sfreq <= 0:
            raise ValueError(f"Error: La frecuencia de muestreo '{sfreq}' no puede ser igual o menor a cero.")
        
        ########## Validación de "info" ##########
        
        if info is not None:
            if sfreq != info.sfreq:
                raise ValueError(f"La frecuencia de muestreo no coincide con la del objeto Info: {sfreq} != {info.sfreq}")
        
            if len(info.ch_names) != self.n_channels:
                raise ValueError(f"Cantidad de canales en 'data' ({self.n_channels}) no coincide con Info ({len(info.ch_names)}).")
        
        ########## Validación de "first_samp" ##########
        
        if not (0 <= first_samp < self.n_samples):
            raise ValueError(f"'first_samp' debe estar entre 0 y {self.n_samples - 1}.")
        
        ########## Validación de "anotaciones" ##########
        
        if anotaciones is not None:
            
            events_duration = np.array(anotaciones.onset) + np.nan_to_num(anotaciones.duration, nan=0.0)
            
            signal_duration = self.n_samples / sfreq
           
            if np.any(events_duration > signal_duration):
                raise ValueError("Al menos una anotación excede la duración total de la señal.")
        
        self.sfreq = sfreq
        self.info = info
        self.first_samp = first_samp
        self.anotaciones = anotaciones

    def get_data(self, picks= None, start:float= 0.0, stop:float= None, reject:float= None, times:bool= False) -> np.ndarray:
        """Obtiene muestras de la señal en un rango temporal dado.

        Parameters:
            picks (str or int or array_like, optional): Canales (nombres o índices) a extraer (por defecto es None, que significa que se seleccionan todos los canales).
            start (float or int, optional): Tiempo inicial en segundos para extraer muestras (por defecto es 0.0).
            stop (float or int, optional): Tiempo final en segundos para extraer muestras (por defecto es None, que significa hasta el final de la señal).
            reject (float or int, optional): Valor pico a pico de umbral para rechazar canales. Si una muestra supera este umbral, el canal se descarta.
            times (bool, optional): Si "True" se retorna también el vector de tiempos asociado a las muestras.

        Returns:
            selected_data (np.ndarray): Matriz con los datos seleccionados (n_canales_sel x n_muestras_sel).
            times_vector (np.ndarray): Vector de tiempos (solo si "times= True").
        
        Raises:
            ValueError: Si los índices seleccionados están fuera de rango."""
            
        check_types( (picks, (str, int, list, type(None))), (start, (int, float)), (stop, (int, float, type(None))), (reject, (int, float, type(None))), (times, bool) )
        
        ########## Selección de canales ##########
        
        if picks is None:
            ########## Caso 1: No se pasó nada → se seleccionan todos los canales ##########
            index_channels = np.arange(self.n_channels)

        else:
            if isinstance(picks, (int, str)):       # Convierte a lista si se pasó un único valor
                picks = [picks]

            ########## Caso 2: ÍNDICES (int o lista de int) ##########
            if all(isinstance(index, int) for index in picks):
                
                index_channels = np.array(picks, dtype=int)
                
                if np.any((index_channels < 0) | (index_channels >= self.n_channels)):  # Comparación para todos los elementos a la vez (Operaciones vectorizadas con NumPy arrays se usa "|")
                    raise ValueError("Algunos índices de canal están fuera de rango.")

            ########## Caso 3 : NOMBRES (str o lista de str) ##########
            elif all(isinstance(name, str) for name in picks):
                
                if self.info is None:
                    raise ValueError("El atributo 'info' es requerido (no puede ser None).")

                index_channels = []
                
                for name in picks:
                    
                    if name not in self.info["ch_names"]:
                        raise ValueError(f"Canal '{name}' no existe en 'info.ch_names'.")
                    
                    index_channels.append(self.info["ch_names"].index(name))

                index_channels = np.array(index_channels, dtype=int)

            else:
                raise TypeError("Todos los elementos de 'picks' deben ser del mismo tipo: 'int' o 'str'.")

        ########## Conversión de tiempos a muestras ##########
        
        s0 = self.first_samp + int(start * self.sfreq)
        
        if stop is None:
            s1 = self.n_samples 
        
        else:
            s1= self.first_samp + int(stop * self.sfreq)
        
        if not (0 <= s0 < s1 <= self.n_samples):
            raise ValueError("Valores de start/stop fuera de rango de la señal.")
 
        ########## Obtencion de muestras de la señal (index_channels) en un rango temporal [s0:s1] ##########
        
        selected_data = self.data[index_channels, s0:s1]

        ########## Rechazo por amplitud ##########
            
        if reject is not None:
            
            if reject <= 0:
                raise ValueError("El parámetro 'reject' debe ser mayor a cero.")
            
            max_vals = np.max(selected_data, axis=1)  
            min_vals = np.min(selected_data, axis=1)  
            
            ptp = max_vals - min_vals              
            
            mask = ptp <= reject                 
            
            selected_data = selected_data[mask, :]         

        ########## Vector de tiempos ##########
        
        if times == True:
            times_vector = (np.arange(start= s0, stop= s1) - self.first_samp) / self.sfreq
            
            return selected_data, times_vector
        
        return selected_data

    def drop_channels(self, ch_names:list) -> "RawSignal":
        """Elimina uno o más canales a partir de ch_names.
        
        Parameter:
            ch_names (str, array_like): Nombres de canales a eliminar.
        
        Returns:
            Objeto RawSignal: Nueva instancia de "RawSignal" con los canales especificados eliminados."""
        
        if self.info is None:
            raise ValueError("El atributo 'info' es requerido (no puede ser None).")
        
        check_types( (ch_names, (str, list)) )
        
        ########## Validación de existencia de los canales ##########
        
        if isinstance(ch_names, str):
            ch_names = [ch_names]
                
        invalid_channels = list(set(ch_names) - set(self.info.ch_names))

        if invalid_channels != []:
            raise ValueError(f"Error: Los siguientes canales no existen en el objeto: {invalid_channels}")

        ########## Índices de canales a eliminar ##########

        to_drop = [self.info.ch_names.index(channel) for channel in ch_names]
        
        ########## Índices de canales a conservar ##########

        mask = [i for i in range(self.n_channels) if i not in to_drop]

        ########## Creación de nuevos datos y copias ##########

        new_data = self.data[mask, :]
        
        new_info = deepcopy(self.info)
        new_annotations = deepcopy(self.anotaciones)

        ########## Actualización de "ch_names", "ch_types" y "bads" ##########

        new_info._data["ch_names"] = [self.info.ch_names[i] for i in mask]
        new_info._data["ch_types"] = [self.info.ch_types[i] for i in mask]

        new_info._data["bads"] = list(set(self.info.bads) - set(ch_names))

        return RawSignal(data= new_data, sfreq= self.sfreq, info= new_info, anotaciones= new_annotations, first_samp= self.first_samp)

    def crop(self, tmin:float= 0.0, tmax:float= None) -> "RawSignal":
        """Obtiene un trozo (Crop) de RawSignal. 
           Limita los datos dentro de RawSignal para obtener un nuevo objeto RawSignal pero con una cantidad de muestras recortadas.
           El parámetro "first_samp" se configura adecuadamente.
        
        Parameter:
            tmin (float, optional): Tiempo inicial, en segundos, para iniciar el recorte (por defecto es 0.0).
            tmax (float or None, optional): Tiempo final, en segundos, para finalizar el recorte (por defecto es None).
        
        Returns:
            Objeto RawSignal: Nueva instancia de "RawSignal" que contiene el segmento temporal recortado.
        
        Raises:
            ValueError: Si los tiempos "tmin" o "tmax" están fuera del rango de la señal."""
                    
        check_types( (tmin, float), (tmax, (float, type(None))) )            
                    
        ########## Conversión de tiempos a muestras ##########
        
        s0 = self.first_samp + int(tmin * self.sfreq)
        
        if tmax is None:
            s1 = self.n_samples
        
        else:
            s1= self.first_samp + int(tmax * self.sfreq)
        
        if not (0 <= s0 < s1 <= self.n_samples):
            raise ValueError("Valores de tmin/tmax fuera de rango de la señal.")

        ########## Extracción ##########
        
        new_data = self.data[:, s0:s1]

        ########## Ajustar annotations ##########
        
        if self.anotaciones is not None:
            
            old_df = self.anotaciones.get_annotations()
            
            # Filtrando los eventos que empiezan entre el tmin y el tmax
            filtered_df = old_df[(old_df["onset"] >= tmin) & (old_df["onset"] < tmax)]
            
            # Ajusta los tiempos de inicio (onset).
            filtered_df["onset"] = filtered_df["onset"] - tmin
            
            new_anots = Anotaciones(onset= filtered_df["onset"].tolist(), duration= filtered_df["duration"].tolist(), description= filtered_df["description"].tolist())
        
        else:
            new_anots = None
        
        return RawSignal(new_data, self.sfreq, info= deepcopy(self.info), anotaciones= new_anots, first_samp= 0)
        
    def describe(self) -> pd.DataFrame:
        """Devuelve un análisis estadístico por canal.
           Para cada canal registrado, se calculan estadísticas como el mínimo, primer cuartil (Q1), mediana, tercer cuartil (Q3) y máximo.

        Returns:
            pd.DataFrame: DataFrame con estadísticas por canal. Cada fila representa un canal con su nombre, tipo y valores estadísticos.

        Raises:
            ValueError: Si la información de canales ("info") no está disponible."""

        if self.info is None:
            raise ValueError("El atributo 'info' es requerido (no puede ser None).")
        
        df = pd.DataFrame(self.data.T, columns=self.info["ch_names"])
        
        return df.describe().T
    
    def filter(self, l_freq:float, h_freq:float, notch_freq:float= 50.0, order:int= 4, fir_window:str= "hamming", quality_factor:float = 30.0) -> "RawSignal":
        """Aplica un filtro pasabanda y un filtro notch a la señal fisiológica.
           
           Parameters:
                l_freq (float): Frecuencia de corte baja (Hz) para el filtro pasabanda.
                h_freq (float): Frecuencia de corte alta (Hz) para el filtro pasabanda.
                notch_freq (float, optional): Frecuencia del filtro notch para eliminar ruido (por defecto 50 Hz).
                order (int, optional): Orden del filtro (por defecto 4).
                fir_window (str, optional): Tipo de ventana para el diseño del filtro FIR (por defecto "hamming").
                quality_factor (float, optional): Factor de calidad del notch (por defecto 30.0).
                
            Returns:    
                RawSignal: Nueva instancia de "RawSignal" con los datos filtrados.
            
            Raises:
                ValueError: Si los valores de "l_freq" o "h_freq" no son válidos.
                ValueError: Si el valor de "notch_freq" no es positivo."""
        
        check_types( (l_freq, float), (h_freq, float), (notch_freq, float), (order, int), (fir_window, str) )
        
        nyq_freq = 0.5 * self.sfreq  # Frecuencia de Nyquist
        
        if not (0 < l_freq < h_freq < nyq_freq):
            raise ValueError("Frecuencias 'l_freq' y 'h_freq' deben ser positivas, con 'l_freq' < 'h_freq' < 'nyq_freq'.")

        if not (0 < notch_freq < nyq_freq):
            raise ValueError("La frecuencia 'notch' debe ser mayor que 0 y menor que 'Nyquist'.")

        if order < 1:
            raise ValueError("El orden del filtro debe ser un entero positivo.")

        # ---------- Filtro notch ---------- #
        b_notch, a_notch = iirnotch(w0= notch_freq, Q= quality_factor, fs= self.sfreq)  # "wo" frecuencia a quitar, "Q" factor de calidad (que tan estrecha es la eliminación), "fs" frecuencia de muestreo, "b_notch" y "a_notch" son los coeficientes del filtro.
        notch_filtered = filtfilt(b_notch, a_notch, self.data, axis= 1)                 # "filtfilt()" aplica el filtro de manera adelante y hacia atrás para evitar el desfase de la señal
        
        # ---------- Filtro pasabanda Butterworth ---------- #
        b_band, a_band = butter(order, [l_freq, h_freq], btype= "band", fs= self.sfreq) # "order" indica la pendiente del filtro, "[l_freq, h_freq]"" rango de frecuencias a quitar, "fs" frecuencia de muestreo
        signal_filtered = filtfilt(b_band, a_band, notch_filtered, axis= 1)             # Aplica el filtro pasabanda sobre la señal ya pasada por el filtro notch

        return RawSignal(data= signal_filtered, sfreq= self.sfreq, info= deepcopy(self.info), anotaciones= deepcopy(self.anotaciones), first_samp= self.first_samp)

    def pick(self, picks) -> "RawSignal":
        """Retorna un subset de canales seleccionados.
        
        Parameters:
            picks (str | array_like | slice): Canales a seleccionar.
                Puede ser:
                - "str": Nombre de un solo canal.
                - "list[str]" : Lista de nombres de canales.
                - "list[int]" : Lista de índices de canales.
                - "slice" : Con los índices de canales.
        
        Returns:
            RawSignal: Nueva instancia de "RawSignal" con el subset de canales elegidos.
        
        Raises:
            ValueError: Si el canal especificado no existe.
            ValueError: Si el índice está fuera del rango de canales."""
        
        if self.info is None:
            raise ValueError("El atributo 'info' es requerido (no puede ser None).")
        
        check_types( (picks, (str, list, slice)) )
        
        all_names = self.info.ch_names

        ########## Caso 1: "pick" es un string ##########

        if isinstance(picks, str):
            
            if picks not in all_names:
                raise ValueError(f"Canal '{picks}' no existe.")
            
            selected_channels = [all_names.index(picks)]

        ########## Caso 2: "pick" es un slice  ##########

        elif isinstance(picks, slice):
            selected_channels = list(range(len(all_names)))[picks]

        else:
            selected_channels = []

            ########## Caso 3: lista de nombres de canales ##########
            
            if all(isinstance(channel, str) for channel in picks):
                invalid_channels = list(set(picks) - set(all_names))
                
                if invalid_channels != []:
                    raise ValueError(f"Canales no encontrados: {invalid_channels}")
                
                selected_channels = [all_names.index(name) for name in picks]

            ########## Caso 4: lista de índices ##########
            
            elif all(isinstance(channel, int) for channel in picks):
                
                for i in picks:
                    if i < 0 or i >= len(all_names):
                        raise ValueError(f"Índice de canal {i} fuera de rango.")
                
                selected_channels = picks

            else:
                raise TypeError("La lista de 'picks' debe contener solo textos (strings) o solo enteros (int).")

        ########## Creación de nuevos datos, copias y actualización de "ch_names" y "ch_types" ##########
        
        new_data = self.data[selected_channels, :]

        new_info = deepcopy(self.info)
        
        new_info._data["ch_names"] = [all_names[i] for i in selected_channels]
        new_info._data["ch_types"] = [self.info.ch_types[i] for i in selected_channels]

        return RawSignal(data= new_data, sfreq= self.sfreq, info= new_info, anotaciones= deepcopy(self.anotaciones), first_samp= self.first_samp)
    
    def set_anotaciones(self, anotaciones:Anotaciones) -> None:
        """Asocia un objeto de tipo "Anotaciones" a la señal fisiológica.
        
        Parameters:
            anotaciones (Anotaciones): Objeto de la clase "Anotaciones" que contiene la información de los eventos.
        
        Raises:
            ValueError: Si alguna anotación está fuera del rango temporal de la señal."""
        
        check_types( (anotaciones, Anotaciones) )
                
        max_time = (self.n_samples - self.first_samp) / self.sfreq
        
        ########## Verificar que el evento completo (onset + duración) no exceda la duración total de la señal ##########
        
        events_duration = np.array(anotaciones.onset) + np.nan_to_num(anotaciones.duration, nan=0.0)
            
        if np.any(events_duration > max_time):
            raise ValueError("Al menos una anotación excede la duración total de la señal.")
        
        self.anotaciones = anotaciones
        
    def plot(self, picks=None, start:float = 0.0, duration:float = 10.0, show_anotaciones:bool = True, legend:bool= False):
        """Grafica un segmento de la señal fisiológica.
        
        Parameters:
            picks (str | int | list of str | list of int | None, opcional): Canal(es) a visualizar.
                Puede ser:
                - str: Nombre de un canal.
                - int: Índice de un canal.
                - list[str]: Lista de nombres de canales.
                - list[int]: Índices de canales.
                - Si es None, se grafican todos los canales.
            start (float, opcional): Tiempo inicial en segundos desde donde comenzar la visualización (por defecto es 0.0).
            duration (float, opcional): Duración del segmento de señal a mostrar en segundos (por defecto es 10.0).
            show_anotaciones (bool, opcional): Si es True, se muestran las anotaciones sobre la señal (por defecto es True).
            legend (bool, opcional): Si es True, se muestran las leyendas de los canales seleccionados.
            
        Raises:
            ValueError: Si el rango temporal excede los límites de la señal."""

        check_types( (picks, (str, int, list, type(None))), (start, float), (duration, float), (show_anotaciones, bool) )

        ########## Extraer segmento ##########
        
        stop = start + duration
        signal_duration = self.n_samples / self.sfreq

        ########## Validación de rango temporal ##########

        if start < 0 or stop > signal_duration:
            raise ValueError(f"El rango solicitado [{start}, {stop}]s excede la duración total de la señal ({signal_duration}s).")

        ########## Obtener datos y vector de tiempos ##########
        if show_anotaciones == True:
            selected_data, times_vector = self.get_data(picks=picks, start=start, stop=stop, times=True)
        
        else:
            selected_data = self.get_data(picks=picks, start=start, stop=stop, times=False)
            times_vector = np.arange(int(start * self.sfreq), int(stop * self.sfreq)) / self.sfreq

        ########## Obtener nombres de canales seleccionados ##########
        
        if picks is None:
            ch_names = self.info["ch_names"]
        
        else:
            ch_names = self.pick(picks).info["ch_names"]

        n_channels = selected_data.shape[0]

        ########## ========== GRAFICAR CON PLOTLY ========== ##########

        fig = make_subplots(rows= n_channels, cols= 1, shared_xaxes= True, vertical_spacing= 0.08, subplot_titles= [f"{name}" for name in ch_names])    # Crea los subplots cada canal donde va la señal  

        for channel_idx in range(n_channels):
            
            signal = go.Scatter(x= times_vector, y= selected_data[channel_idx], mode= "lines", name= ch_names[channel_idx], line= dict(width=1))   # Grafica la señal ("name" es para la leyenda, "line" es el grosor del trazo)
            
            fig.add_trace(signal, row= channel_idx+1, col= 1)    # Añade la señal al subplot del canal correspondiente

        ########## Añadir anotaciones si corresponde ##########
        
        if show_anotaciones and self.anotaciones is not None:
            
            events = self.anotaciones.get_annotations()
            
            for _, event in events.iterrows():
                
                onset = event["onset"]
                dur = event["duration"]
                desc = event["description"]

                if start <= onset <= stop:
                    
                    for channel_idx in range(n_channels):
                       
                        fig.add_vline(x= onset, line_dash= "dot", line_color= "red", row= channel_idx+1, col= 1)   # Añade la linea vertical punteada que indica el inicio del evento
                        
                        if dur > 0:
                            fig.add_vrect(x0= onset, x1= onset + dur, fillcolor= "red", opacity= 0.2, line_width= 0, row= channel_idx+1, col= 1) # Añade el sombreado coloreado que indica la duracion del evento
                        
                        event_text= go.Scatter(x= [onset], y= [max(selected_data[channel_idx])], text= [desc], mode= "text", textposition= "top right") # Añade un texto que indica la descripcion del evento
                        
                        fig.add_trace(event_text, row=channel_idx+1, col=1)  # Añade la descripcion al subplot

        ########## Personalización de la figura ##########
        
        fig.update_layout(height= 300 + 250 * n_channels, title_text= "Señal (interactivo)", margin= dict(t=60, b=60), showlegend= legend)
        fig.update_xaxes(title_text="Tiempo (s)")
        fig.update_yaxes(title_text="Amplitud (uV o mV)")
        fig.show()
        
    def signal_spectrogram(self, channel= 0, nperseg= 256, noverlap= 128):
        """Grafica el espectrograma tiempo-frecuencia de un canal.
        
        Parameters:
            channel (int | str): Canal a graficar (índice o nombre). (por defecto es 0).
            nperseg (int): Cantidad de muestras por segmento para el cálculo del espectrograma, afecta la resolución temporal y espectral (por defecto es 256).
            noverlap (int): Cantidad de muestras que se solapan entre segmentos consecutivos (por defecto es 128)."""

        check_types((channel, (int, str)), (nperseg, int), (noverlap, int))

    # Obtener canal
        if isinstance(channel, str):
            if self.info is None or channel not in self.info["ch_names"]:
                raise ValueError(f"Canal '{channel}' no encontrado.")
            
            channel_idx = self.info["ch_names"].index(channel)
           
            name = channel
        
        else:
            if channel < 0 or channel >= self.n_channels:
                raise ValueError(f"El canal {channel} está fuera de rango.")
            
            channel_idx = channel
            
            name = self.info["ch_names"][channel] if self.info else f"Canal {channel}"

        signal = self.data[channel_idx]
        
        # Normalización (media 0, varianza 1)
        signal = normalize_signal(signal)
        
        # ---------- Calcular espectrograma ---------- #
        frecuencies, times, spectral_density_power = spectrogram(signal, fs= self.sfreq, nperseg= nperseg, noverlap= noverlap)
        
        # ---------- Crear figura interactiva con plotly ---------- #
        heatmap = go.Heatmap(z= 10 * np.log10(spectral_density_power + 1e-12), x= times, y= frecuencies, colorscale= "Viridis", colorbar= dict(title="Potencia (dB)")) # "z" Convierte la densidad espectral de potencia a decibeles (dB) para mejorar la visualización
        
        fig = go.Figure(data= heatmap)

        fig.update_layout(title= f"Espectrograma - {name}", xaxis_title= "Tiempo (s)", yaxis_title= "Frecuencia (Hz)", height= 500, margin= dict(t=60, b=60))

        fig.show()

    def hilbert(self, channels= None):
        """Grafica la señal original y la envolvente de Hilbert para uno o más canales.

        Parameters:
            channels (list): Índices de los canales a graficar. Si es None, grafica todos."""
            
        check_types((channels, list))

        # ---------- Selección de canales ---------- #
        if channels is None:
            channels= range(self.n_channels)
        
        n_channels = len(channels)
        
        # ---------- Eje de tiempo ---------- #
        times = np.arange(self.n_samples) / self.sfreq

        # ---------- Crear figura interactiva ---------- #
        fig = make_subplots(rows= n_channels, cols= 1, shared_xaxes= True, vertical_spacing= 0.08, subplot_titles= [self.info["ch_names"][ch] if self.info else f"Canal {ch}" for ch in channels])

        # ---------- Agregar gráfico por canal ---------- #
        for channel_idx, channel in enumerate(channels):
            
            signal = self.data[channel]
            
            hilbert_envelope = np.abs(hilbert(signal)) # Transformada de Hilbert

            # Señal original
            tr_señal = go.Scatter(x= times, y= signal, mode= "lines", name= f"Señal {self.info['ch_names'][channel]}", line= dict(color= "blue", width= 1))
            
            # Envolvente
            trace_envelope = go.Scatter(x= times, y= hilbert_envelope, mode= "lines", name= f"Envolvente {self.info['ch_names'][channel]}", line=dict(color= "red", width= 2, dash= "dot"))

            fig.add_trace(tr_señal, row= channel_idx+1, col= 1)
            fig.add_trace(trace_envelope, row= channel_idx+1, col= 1)
        
        # ---------- Configuración general del gráfico ---------- #
        fig.update_layout(height= 300 + 250 * n_channels, title_text= "Señal y Envolvente de Hilbert", margin= dict(t=60, b=60), showlegend= True)
        fig.update_xaxes(title_text="Tiempo (s)")
        fig.update_yaxes(title_text="Amplitud (uV o mV)")
        fig.show()                   

    def __getitem__(self, key:str):
        """Permite acceder a las muestras de la señal como si fuera un diccionario.
        
        Parameters:
            key (str): Clave a acceder.
            
        Raises:
            KeyError: Si la clave no existe en el objeto."""
        
        # Caso 1: solo nombre de canal
        if isinstance(key, str):
           try:
                return self.get_data(picks=key, times=True)
           except ValueError:
                raise KeyError(f"Canal '{key}' no existe")
        
        # Caso 2: acceso por canal
        elif isinstance(key, tuple) and len(key) == 2:
            picks, time_sel = key
            
            # validar picks
            if not isinstance(picks, (str, list)):
                raise KeyError(f"Picks inválido: {picks}")
            
            # interpretar slice
            if isinstance(time_sel, slice):
                start_sample = time_sel.start or 0
                stop_sample  = time_sel.stop  or self.n_samples
                start = start_sample / self.sfreq
                stop  = stop_sample  / self.sfreq
            else:
                raise KeyError(f"Rango de tiempo inválido: {time_sel}")

            try:
                return self.get_data(picks=picks, start=start, stop=stop, times=True)
            except ValueError:
                raise KeyError(f"Canal o rango inválido: {key}")
        
        # Cualquier otra clave no soportada
        else:
            raise KeyError(f"Clave inválida: {key}")    

class EEGSignal(RawSignal):
    """ Clase para señales de Electroencefalografía (EEG), con análisis de referencia y frecuencia. """

    def __init__(self, data:np.ndarray, sfreq:int, info:Info= None, first_samp:int= 0, anotaciones:Anotaciones= None, reference:str= "desconocida"):
        """ Inicializa la instancia de la clase EEGSignal.

        Parámetros:
            data (np.ndarray): Datos de la señal EEG (n_canales, n_muestras).
            sfreq (int): Frecuencia de muestreo de la señal en Hz.
            info (Info, opcional): Objeto de información adicional sobre los canales.
            first_samp (int, opcional): Índice de la primera muestra. 
            anotaciones (Anotaciones, opcional): Objeto de anotaciones para eventos en la señal.
            reference (str, opcional): Tipo de referencia utilizada en la señal (por defecto es "desconocida")."""
            
        super().__init__(data, sfreq, info, first_samp, anotaciones)
        
        check_types((reference, str))
        
        self.reference = reference

    def set_reference_to(self, channel):
        """ Cambia la referencia de la señal a un canal específico. 
        La señal de cada canal se ajusta restando el valor del canal de referencia a todos los canales.

        Parámetros:
            channel (str | int): Nombre del canal o índice del canal a utilizar como referencia.

        Excepciones:
            ValueError: Si el nombre del canal no existe.
            IndexError: Si el índice del canal está fuera de rango.
            TypeError: Si el canal no es ni un nombre (str) ni un índice (int)."""
        
        check_types((channel, (int, str)))
            
        if isinstance(channel, str):
            if channel not in self.info["ch_names"]:
                raise ValueError(f"Canal '{channel}' no existe en los canales disponibles.")
            
            idx = self.info["ch_names"].index(channel)
            
        elif isinstance(channel, int):
            if channel < 0 or channel >= self.n_channels:
                raise IndexError(f"Índice de canal fuera de rango: {channel}")
            
            idx = channel
        
        else:
            raise TypeError("El canal debe ser un nombre (str) o un índice (int).")

        reference = self.data[idx, :]
        self.data = self.data - reference
        self.reference = f"referenciado a {self.info['ch_names'][idx]}"

    def apply_laplacian(self):
        """Aplica un filtro Laplaciano simple a los datos de la señal EEG. 
        El filtro se calcula restando el promedio de los canales vecinos inmediatos del canal central.

        Excepciones:
            ValueError: Si hay menos de 3 canales, no es posible aplicar el filtro Laplaciano."""
        
        if self.n_channels < 3:
            raise ValueError("Se necesitan al menos 3 canales para aplicar Laplaciano.")

        lap_data = np.zeros_like(self.data)

        # Aplicar el filtro Laplaciano a los canales internos
        for i in range(1, self.n_channels - 1):
            vecinos = (self.data[i - 1, :] + self.data[i + 1, :]) / 2
            lap_data[i, :] = self.data[i, :] - vecinos

        # Mantener los bordes sin modificar
        lap_data[0, :] = self.data[0, :]
        lap_data[-1, :] = self.data[-1, :]

        self.data = lap_data
        self.reference = "laplaciano"

    def plot_welch(self, channels:list= None) -> dict:
        """Calcula la potencia promedio en las bandas de frecuencia clásicas de EEG para uno o más canales.

        Las bandas consideradas son:
        - Delta: 0.5 - 4 Hz
        - Theta: 4 - 8 Hz
        - Alpha: 8 - 13 Hz
        - Beta: 13 - 30 Hz
        - Gamma: 30 - 45 Hz

        Parámetros:
            channels (list | None, opcional): Lista de canales a analizar. Si es None, se utilizan todos los canales.

        Retorna:
            dict: Diccionario con la potencia promedio por banda para cada canal. Las claves son los nombres de los canales, y los valores son diccionarios con las potencias de las bandas.

        Excepciones:
            ValueError: Si los canales no están disponibles en la señal."""
        
        check_types((channels, (list, None)))
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta':  (13, 30),
            'gamma': (30, 45)
        }

        # Corregir: si canales es None, usar todos los canales
        channels = list(range(self.n_channels)) if channels is None else channels

        # Obtener los datos correspondientes
        signals = self.get_data(picks=channels)

        # Obtener nombres correctos de los canales
        ch_names = self.pick(channels).info["ch_names"]

        results = {}

        for i, signal in enumerate(signals):
            
            f, Pxx = welch(signal, fs=self.sfreq, nperseg=1024)
            
            powers = {}
            
            for band_name, (fmin, fmax) in bands.items():
                idx = (f >= fmin) & (f <= fmax)
                powers[band_name] = np.trapezoid(Pxx[idx], f[idx])  # Integración por trapecio
            
            results[ch_names[i]] = powers

        return results

    def fourier(self, channel: str):
        """Grafica interactiva del espectro de Fourier de un canal.

        Parámetros:
            channel (str): Nombre del canal a analizar.

        Raises:
            ValueError: Si el canal no se encuentra en los nombres definidos en Info.
        """
        check_types((channel, str))  # Validar tipo

        # ---------- Validación del canal ---------- #
        if self.info is None or "ch_names" not in self.info:
            raise ValueError("El atributo 'info' debe estar definido y contener 'ch_names'.")

        if channel not in self.info["ch_names"]:
            raise ValueError(f"Canal '{channel}' no encontrado en Info.")

        # ---------- Selección de la señal del canal ---------- #
        idx = self.info["ch_names"].index(channel)
        x = self.data[idx]
        N = len(x)
        T = 1.0 / self.sfreq

        # ---------- Transformada de Fourier ---------- #
        yf = fft(x)
        xf = fftfreq(N, T)[:N // 2]
        amplitud = 2.0 / N * np.abs(yf[:N // 2])

        # ---------- Crear gráfico interactivo ---------- #
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=xf, y=amplitud,
            mode="lines",
            name=f"Espectro - {channel}",
            line=dict(color="darkblue")
        ))

        # ---------- Configurar diseño ---------- #
        fig.update_layout(
            title=f"Espectro de Fourier - Canal {channel}",
            xaxis_title="Frecuencia (Hz)",
            yaxis_title="Amplitud",
            template="plotly_white",
            height=400
        )

        fig.update_xaxes(range=[0, 60])

        fig.update_yaxes(type='log')

        fig.show()


class ECGSignal(RawSignal):
    """Clase para señales de Electrocardiografía (ECG), con métodos para detectar picos R y calcular la frecuencia cardíaca y la variabilidad de la frecuencia cardíaca (HRV)."""

    def __init__(self, data:np.ndarray, sfreq:int, info:Info= None, first_samp:int= 0, anotaciones:Anotaciones= None):
        """Inicializa una instancia de la clase ECGSignal.
        
        Parámetros:
            data (np.ndarray): Array de datos de la señal ECG (n_canales, n_muestras).
            sfreq (int): Frecuencia de muestreo en Hz.
            info (Info, opcional): Objeto con información adicional sobre los canales.
            first_samp (int, opcional): Índice de la primera muestra de la señal a usar.
            anotaciones (Anotaciones, opcional): Objeto con anotaciones de eventos."""
            
        super().__init__(data, sfreq, info, first_samp, anotaciones)
        
        self.r_peaks = None
        self.heart_rate_bpm = None

    def detect_r_peaks(self, channel= 0, umbral= 0.5) -> np.ndarray:
        """Detecta los picos R de la señal de ECG en el canal indicado.
        
        Parámetros:
            channel (int, opcional): El índice del canal en el que detectar los picos R. El valor por defecto es 0.
            umbral (float, opcional): El valor mínimo de amplitud para considerar un pico R. El valor por defecto es 0.5.
        
        Retorna:
            np.ndarray: Array con los índices de los picos R detectados en el canal.
        
        Excepciones:
            ValueError: Si el umbral es menor o igual a cero.
            IndexError: Si el canal indicado está fuera del rango de canales."""
            
        check_types((channel, int), (umbral, float))    
            
        # Validación de umbral
        if umbral <= 0:
            raise ValueError(f"El umbral debe ser un valor positivo. Recibido: {umbral}")

        # Validación del canal
        if channel < 0 or channel >= self.n_channels:
            raise IndexError(f"Índice de canal fuera de rango: {channel}")

        signal = self.data[channel, :]
        
        peaks, _ = find_peaks(signal, height= umbral, distance= int(0.3 * self.sfreq))
        
        self.r_peaks = peaks
        
        return peaks

    def heart_rate(self) -> float:
        """Calcula la frecuencia cardíaca (BPM) a partir de los picos R detectados.
        
        Retorna:
            float: La frecuencia cardíaca calculada en latidos por minuto (BPM).
        
        Excepciones:
            ValueError: Si no se han detectado picos R antes de llamar a este método."""
            
        if self.r_peaks is None:
            raise ValueError("Primero debe ejecutar el método detectar_picos_r().")

        rr_intervals = np.diff(self.r_peaks) / self.sfreq  
                
        self.heart_rate_bpm = 60 / np.mean(rr_intervals)
        
        return self.heart_rate_bpm

    def calc_hrv(self) -> pd.DataFrame:
        """Calcula la variabilidad de la frecuencia cardíaca (HRV) a partir de los picos R detectados.
        
        Retorna:
            pd.DataFrame: Un DataFrame con las métricas de HRV:
                - SDNN (s): Desviación estándar de los intervalos RR.
                - RMSSD (s): Raíz cuadrada de la media de los cuadrados de las diferencias RR.
                - pNN50 (%): Porcentaje de intervalos RR consecutivos que difieren más de 50 ms.
                - RR Mean (s): Promedio de los intervalos RR.
        
        Excepciones:
            ValueError: Si no se han detectado picos R antes de llamar a este método."""
            
        if self.r_peaks is None:
            raise ValueError("Primero debe ejecutar detectar_picos_r() para poder calcular HRV.")
        
        rr_intervals = np.diff(self.r_peaks) / self.sfreq  # en segundos
        
        # Cálculo de las métricas HRV
        sdnn = np.std(rr_intervals)  # Desviación estándar de los intervalos RR
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # RMSSD
        pnn50 = np.sum(np.diff(rr_intervals) > 0.05) / len(rr_intervals) * 100  # PNN50
        rr_mean = np.mean(rr_intervals)  # RR mean

        # Retornar los resultados
        
        return pd.DataFrame([{
            "RMSSD (s)": rmssd,
            "SDNN (s)": sdnn,
            "pNN50 (%)": pnn50,
            "RR Mean (s)": rr_mean,
            }])

    def plot_peaks(self, channel=0):
        """Grafica la señal de ECG con los picos R detectados sobreimpresos.
        
        Parámetros:
            channel (int, opcional): El índice del canal a graficar. El valor por defecto es 0.
        
        Excepciones:
            ValueError: Si no se han detectado picos R antes de llamar a este método."""
        
        if self.r_peaks is None:
            raise ValueError("Debe ejecutar el método detectar_picos_r() antes.")

        signal = self.data[channel, :]
        
        times = np.arange(len(signal)) / self.sfreq

        name = self.info["ch_names"][channel] if self.info else f"Canal {channel}"

        # ---------- Crear gráfico interactivo ---------- #
        fig = go.Figure()

        # Trazar la señal ECG
        fig.add_trace(go.Scatter(x= times, y= signal, mode= 'lines', name= f"ECG - {name}", line= dict(color='royalblue')))

        # Trazar los picos R
        fig.add_trace(go.Scatter(x= times[self.r_peaks], y= signal[self.r_peaks], mode= 'markers', name= "Picos R", marker= dict(color= 'red', size= 8, symbol= 'circle'), hovertemplate= "Tiempo: %{x:.2f}s<br>Amplitud: %{y:.2f}"))

        # ---------- Configuración del gráfico ---------- #
        fig.update_layout(title= "ECG con Picos R detectados", xaxis_title= "Tiempo (s)", yaxis_title= "Amplitud", legend= dict(x= 0.01, y= 0.99), template= "plotly_white", height= 400)

        fig.show()

class EMGSignal(RawSignal):
    """ Clase para señales de Electromiografía (EMG), con detección de activaciones musculares y análisis energético. """

    def __init__(self, data:np.ndarray, sfreq:int, info:Info= None, first_samp:int= 0, anotaciones:Anotaciones= None, umbral:float= 0.05):
        """Inicializa la instancia de la clase EMGSignal.

        Parámetros:
            data (np.ndarray): Datos de la señal EMG (n_canales, n_muestras).
            sfreq (int): Frecuencia de muestreo de la señal en Hz.
            info (Info, opcional): Objeto de información adicional sobre los canales.
            first_samp (int, opcional): Índice de la primera muestra.
            anotaciones (Anotaciones, opcional): Objeto de anotaciones para eventos en la señal.
            umbral (float, opcional): Umbral de activación en microvoltios (por defecto 0.05)."""
            
        super().__init__(data, sfreq, info, first_samp, anotaciones)
        
        self.umbral_activation = umbral  # Umbral de activación
        self.activations = []  # Lista de activaciones por canal

    def detect_activations(self) -> list:
        """Detecta las activaciones (muestras donde el valor absoluto supera el umbral) en todos los canales.

        Retorna:
            activations (list): Lista de arrays con los índices de las muestras donde se detectan activaciones por canal."""
            
        self.activations = []
        
        for channel in range(self.n_channels):
            activations_ch = np.where(np.abs(self.data[channel]) > self.umbral_activation)[0]
            self.activations.append(activations_ch)
        
        return self.activations

    def calculate_rms_level(self, channels:list= None, window:float= 1.0) -> dict:
        """Calcula el nivel RMS (root mean square) para uno o más canales de EMG en ventanas de tiempo.

        El RMS se calcula como la raíz cuadrada de la media de los cuadrados de las muestras dentro de cada ventana.

        Parámetros:
            channels (list[str] | list[int] | None): Canales a analizar. Si es None, se analizan todos los canales.
            window (float): Duración de la ventana en segundos para calcular el RMS.

        Retorna:
            result (dict): Diccionario donde las claves son los nombres de los canales y los valores son arrays con el valor RMS por ventana.

        Excepciones:
            ValueError: Si la duración de la ventana no es un número positivo."""
            
        check_types((channels, (list, None))), (window, (int, float))   
        
        # Validación de la duración de la ventana
        if not isinstance(window, (int, float)) or window <= 0:
            raise ValueError("La duración de la ventana debe ser un número positivo.")

        # Obtener los datos de los canales seleccionados
        data_channels = self.get_data(picks= channels)

        # Definir el número de muestras por ventana
        samples_per_window = int(self.sfreq * window)

        # Crear un diccionario para almacenar los resultados
        result = {}
        
        ch_names = self.pick(channels).info["ch_names"] if channels is not None else self.info["ch_names"]

        # Calcular el RMS por ventana para cada canal
        for i, signal in enumerate(data_channels):
            
            rms_per_window = []
            
            for start in range(0, len(signal), samples_per_window):
                segmento = signal[start:start + samples_per_window]
                
                if len(segmento) == 0:
                    continue
                
                rms = np.sqrt(np.mean(segmento ** 2))  # Calcular el RMS
                rms_per_window.append(rms)
            
            result[ch_names[i]] = np.array(rms_per_window)

        return result
